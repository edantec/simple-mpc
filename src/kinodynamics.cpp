#include "simple-mpc/kinodynamics.hpp"

namespace simple_mpc {
using namespace aligator;

KinodynamicsProblem::KinodynamicsProblem(const RobotHandler &handler)
    : Base(handler) {}

KinodynamicsProblem::KinodynamicsProblem(const KinodynamicsSettings &settings,
                                         const RobotHandler &handler)
    : Base(handler) {
  initialize(settings);
}

void KinodynamicsProblem::initialize(const KinodynamicsSettings &settings) {
  settings_ = settings;
  nu_ = nv_ - 6 + settings_.force_size * (int)handler_.getFeetNames().size();
  if (nu_ != settings_.u0.size()) {
    throw std::runtime_error("settings.u0 does not have the correct size nu");
  }
  control_ref_ = settings_.u0;

  // Set up cost names used in kinodynamics problem
  std::size_t cost_incr = 0;
  cost_map_.insert({"state_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"control_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"centroidal_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"centroidal_derivative_cost", cost_incr});
  cost_incr++;
  for (auto cname : handler_.getFeetNames()) {
    cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
  cost_incr = 0;
  terminal_cost_map_.insert({"state_cost", cost_incr});
  cost_incr++;
  terminal_cost_map_.insert({"centroidal_cost", cost_incr});
  cost_incr++;
  for (auto &cname : handler_.getFeetNames()) {
    terminal_cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
}

StageModel KinodynamicsProblem::createStage(
    const ContactMap &contact_map,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  auto space = MultibodyPhaseSpace(handler_.getModel());
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  std::vector<std::string> contact_names = contact_map.getContactNames();
  auto contact_poses = contact_map.getContactPoses();

  computeControlFromForces(force_refs);

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));
  auto centder_mom = CentroidalMomentumDerivativeResidual(
      space.ndx(), handler_.getModel(), settings_.gravity, contact_states,
      handler_.getFeetIds(), 6);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, control_ref_, settings_.w_u));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));
  rcost.addCost(QuadraticResidualCost(space, centder_mom, settings_.w_centder));

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
    frame_placement.translation() = contact_poses[i];
    FramePlacementResidual frame_residual = FramePlacementResidual(
        space.ndx(), nu_, handler_.getModel(), frame_placement,
        handler_.getFootId(contact_names[i]));
    int is_active = 0;
    if (!contact_states[i])
      is_active = 1;

    rcost.addCost(QuadraticResidualCost(space, frame_residual,
                                        settings_.w_frame * is_active));
  }

  KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, handler_.getModel(), settings_.gravity, contact_states,
      handler_.getFeetIds(), settings_.force_size);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

void KinodynamicsProblem::setReferencePose(const std::size_t t,
                                           const std::string &ee_name,
                                           const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
  cfr->setReference(pose_ref);
}

void KinodynamicsProblem::setReferencePoses(
    const std::size_t t,
    const std::map<std::string, pinocchio::SE3> &pose_refs) {
  if (pose_refs.size() != handler_.getFeetNames().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }

  CostStack *cs = getCostStack(t);
  for (auto ee_name : handler_.getFeetNames()) {
    QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
        &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
    FramePlacementResidual *cfr =
        dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
    cfr->setReference(pose_refs.at(ee_name));
  }
}

void KinodynamicsProblem::setTerminalReferencePose(
    const std::string &ee_name, const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getTerminalCostStack();
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[terminal_cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
  cfr->setReference(pose_ref);
}

pinocchio::SE3
KinodynamicsProblem::getReferencePose(const std::size_t t,
                                      const std::string &ee_name) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qc->residual_);
  return cfr->getReference();
}

void KinodynamicsProblem::computeControlFromForces(
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  for (std::size_t i = 0; i < handler_.getFeetNames().size(); i++) {
    if (settings_.force_size != force_refs.at(handler_.getFootName(i)).size()) {
      throw std::runtime_error(
          "force size in settings does not match reference force size");
    }
    control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(handler_.getFootName(i));
  }
}

void KinodynamicsProblem::setReferenceForces(
    const std::size_t i,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  computeControlFromForces(force_refs);
  setReferenceControl(i, control_ref_);
}

void KinodynamicsProblem::setReferenceForce(const std::size_t i,
                                            const std::string &ee_name,
                                            const Eigen::VectorXd &force_ref) {
  std::vector<std::string> hname = handler_.getFeetNames();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();
  control_ref_.segment(id * settings_.force_size, settings_.force_size) =
      force_ref;
  setReferenceControl(i, control_ref_);
}

Eigen::VectorXd
KinodynamicsProblem::getReferenceForce(const std::size_t i,
                                       const std::string &ee_name) {
  std::vector<std::string> hname = handler_.getFeetNames();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();

  return getReferenceControl(i).segment(id * settings_.force_size,
                                        settings_.force_size);
}

Eigen::VectorXd
KinodynamicsProblem::getMultibodyState(const Eigen::VectorXd &x_multibody) {
  if (x_multibody.size() != handler_.getState().size()) {
    throw std::runtime_error("x_multibody is of incorrect size");
  }
  handler_.updateInternalData(x_multibody);
  return x_multibody;
}

CostStack KinodynamicsProblem::createTerminalCost() {
  auto ter_space = MultibodyPhaseSpace(handler_.getModel());
  auto term_cost = CostStack(ter_space, nu_);
  auto cent_mom = CentroidalMomentumResidual(
      ter_space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));

  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));
  term_cost.addCost(
      QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent));
  for (auto const &name : handler_.getFeetNames()) {
    FramePlacementResidual frame_residual = FramePlacementResidual(
        ter_space.ndx(), nu_, handler_.getModel(), handler_.getFootPose(name),
        handler_.getFootId(name));

    term_cost.addCost(
        QuadraticResidualCost(ter_space, frame_residual, settings_.w_frame));
  }

  return term_cost;
}

void KinodynamicsProblem::createTerminalConstraint() {
  if (!problem_initialized_) {
    throw std::runtime_error("Create problem first!");
  }
  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ndx_, nu_, handler_.getModel(), handler_.getComPosition());

  StageConstraint term_constraint_com = {com_cstr, EqualityConstraint()};
  problem_->addTerminalConstraint(term_constraint_com);
  terminal_constraint_ = true;
}

void KinodynamicsProblem::updateTerminalConstraint() {
  if (terminal_constraint_) {
    CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
        ndx_, nu_, handler_.getModel(), handler_.getComPosition());

    StageConstraint term_constraint_com = {com_cstr, EqualityConstraint()};
    problem_->removeTerminalConstraints();
    problem_->addTerminalConstraint(term_constraint_com);
  }
}

} // namespace simple_mpc
