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
  nu_ = nv_ - 6 + settings_.force_size * (int)handler_.get_ee_names().size();
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
  for (auto cname : handler_.get_ee_names()) {
    cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
}

StageModel KinodynamicsProblem::create_stage(
    const ContactMap &contact_map,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  std::vector<std::string> contact_names = contact_map.getContactNames();
  auto contact_poses = contact_map.getContactPoses();

  compute_control_from_forces(force_refs);

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.get_rmodel(), Eigen::VectorXd::Zero(6));
  auto centder_mom = CentroidalMomentumDerivativeResidual(
      space.ndx(), handler_.get_rmodel(), settings_.gravity, contact_states,
      handler_.get_ee_ids(), 6);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, control_ref_, settings_.w_u));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));
  rcost.addCost(QuadraticResidualCost(space, centder_mom, settings_.w_centder));

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
    frame_placement.translation() = contact_poses[i];
    FramePlacementResidual frame_residual = FramePlacementResidual(
        space.ndx(), nu_, handler_.get_rmodel(), frame_placement,
        handler_.get_ee_id(contact_names[i]));
    int is_active = 0;
    if (!contact_states[i])
      is_active = 1;

    rcost.addCost(QuadraticResidualCost(space, frame_residual,
                                        settings_.w_frame * is_active));
  }

  KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, handler_.get_rmodel(), settings_.gravity, contact_states,
      handler_.get_ee_ids(), settings_.force_size);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

void KinodynamicsProblem::set_reference_pose(const std::size_t t,
                                             const std::string &ee_name,
                                             const pinocchio::SE3 &pose_ref) {
  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
  cfr->setReference(pose_ref);
}

void KinodynamicsProblem::set_reference_poses(
    const std::size_t t,
    const std::map<std::string, pinocchio::SE3> &pose_refs) {
  if (pose_refs.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }

  CostStack *cs = get_cost_stack(t);
  for (auto ee_name : handler_.get_ee_names()) {
    QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
        &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
    FramePlacementResidual *cfr =
        dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
    cfr->setReference(pose_refs.at(ee_name));
  }
}

pinocchio::SE3
KinodynamicsProblem::get_reference_pose(const std::size_t t,
                                        const std::string &ee_name) {
  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qc->residual_);
  return cfr->getReference();
}

void KinodynamicsProblem::compute_control_from_forces(
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  for (std::size_t i = 0; i < handler_.get_ee_names().size(); i++) {
    if (settings_.force_size != force_refs.at(handler_.get_ee_name(i)).size()) {
      throw std::runtime_error(
          "force size in settings does not match reference force size");
    }
    control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(handler_.get_ee_name(i));
  }
}

void KinodynamicsProblem::set_reference_forces(
    const std::size_t i,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  compute_control_from_forces(force_refs);
  set_reference_control(i, control_ref_);
}

void KinodynamicsProblem::set_reference_force(
    const std::size_t i, const std::string &ee_name,
    const Eigen::VectorXd &force_ref) {
  std::vector<std::string> hname = handler_.get_ee_names();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();
  control_ref_.segment(id * settings_.force_size, settings_.force_size) =
      force_ref;
  set_reference_control(i, control_ref_);
}

Eigen::VectorXd
KinodynamicsProblem::get_reference_force(const std::size_t i,
                                         const std::string &ee_name) {
  std::vector<std::string> hname = handler_.get_ee_names();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();

  return get_reference_control(i).segment(id * settings_.force_size,
                                          settings_.force_size);
}

Eigen::VectorXd
KinodynamicsProblem::get_x0_from_multibody(const Eigen::VectorXd &x_multibody) {
  if (x_multibody.size() != handler_.get_x0().size()) {
    throw std::runtime_error("x_multibody is of incorrect size");
  }
  handler_.updateInternalData(x_multibody);
  return x_multibody;
}

CostStack KinodynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

} // namespace simple_mpc
