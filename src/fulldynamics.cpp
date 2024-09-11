#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/utils/exceptions.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/fulldynamics.hpp"

namespace simple_mpc {
using namespace aligator;

FullDynamicsProblem::FullDynamicsProblem(const RobotHandler &handler)
    : Base(handler) {}

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings &settings,
                                         const RobotHandler &handler)
    : Base(handler) {

  initialize(settings);
}

void FullDynamicsProblem::initialize(const FullDynamicsSettings &settings) {

  settings_ = settings;
  actuation_matrix_.resize(nv_, nu_);
  actuation_matrix_.setZero();
  actuation_matrix_.bottomRows(nu_).setIdentity();

  prox_settings_ = ProximalSettings(1e-9, 1e-10, 1);

  for (auto &name : handler_.get_ee_names()) {
    auto frame_ids = handler_.get_ee_id(name);
    auto joint_ids = handler_.get_rmodel().frames[frame_ids].parentJoint;
    pinocchio::SE3 pl1 = handler_.get_rmodel().frames[frame_ids].placement;
    pinocchio::SE3 pl2 = handler_.get_ee_pose(name);
    pinocchio::RigidConstraintModel constraint_model =
        pinocchio::RigidConstraintModel(pinocchio::ContactType::CONTACT_6D,
                                        handler_.get_rmodel(), joint_ids, pl1,
                                        0, pl2, pinocchio::LOCAL);
    constraint_model.corrector.Kp << 1, 1, 10, 1, 1, 1;
    constraint_model.corrector.Kd << 50, 50, 50, 50, 50, 50;
    constraint_model.name = name;
    constraint_models_.push_back(constraint_model);
  }

  // Set up cost names used in full dynamics problem
  std::size_t cost_incr = 0;
  cost_map_.insert({"state_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"control_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"centroidal_cost", cost_incr});
  cost_incr++;
  for (auto &cname : handler_.get_ee_names()) {
    cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
  for (auto &cname : handler_.get_ee_names()) {
    cost_map_.insert({cname + "_force_cost", cost_incr});
    cost_incr++;
  }
  cost_incr = 0;
  terminal_cost_map_.insert({"state_cost", cost_incr});
  cost_incr++;
  terminal_cost_map_.insert({"centroidal_cost", cost_incr});
  cost_incr++;
  for (auto &cname : handler_.get_ee_names()) {
    terminal_cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
}

StageModel FullDynamicsProblem::create_stage(
    const ContactMap &contact_map,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, settings_.u0, settings_.w_u));

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.get_rmodel(), Eigen::VectorXd::Zero(6));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));

  pinocchio::context::RigidConstraintModelVector cms;
  std::vector<std::string> contact_names = contact_map.getContactNames();
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  if (contact_states.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "contact states size does not match number of end effectors");
  }

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
    frame_placement.translation() = contact_poses[i];
    FramePlacementResidual frame_residual = FramePlacementResidual(
        space.ndx(), nu_, handler_.get_rmodel(), frame_placement,
        handler_.get_ee_id(contact_names[i]));

    int is_active = 0;
    if (contact_states[i])
      cms.push_back(constraint_models_[i]);
    else
      is_active = 1;

    rcost.addCost(QuadraticResidualCost(space, frame_residual,
                                        settings_.w_frame * is_active));
  }

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    std::shared_ptr<ContactForceResidual> frame_force;
    int is_active = 0;
    if (contact_states[i]) {
      frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), handler_.get_rmodel(), actuation_matrix_, cms,
          prox_settings_, force_refs.at(handler_.get_ee_name(i)),
          handler_.get_ee_name(i));
      is_active = 1;
    } else {
      frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), handler_.get_rmodel(), actuation_matrix_,
          constraint_models_, prox_settings_,
          force_refs.at(handler_.get_ee_name(i)), handler_.get_ee_name(i));
    }
    rcost.addCost(QuadraticResidualCost(space, *frame_force,
                                        settings_.w_forces * is_active));
  }

  MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(
      space, actuation_matrix_, cms, prox_settings_);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  StageModel stm = StageModel(rcost, dyn_model);

  // Constraints
  ControlErrorResidual ctrl_fn =
      ControlErrorResidual(space.ndx(), Eigen::VectorXd::Zero(nu_));
  stm.addConstraint(ctrl_fn, BoxConstraint(settings_.umin, settings_.umax));
  StateErrorResidual state_fn = StateErrorResidual(space, nu_, space.neutral());
  std::vector<int> state_id;
  for (int i = 6; i < nv_; i++) {
    state_id.push_back(i);
  }
  FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
  stm.addConstraint(state_slice,
                    BoxConstraint(-settings_.qmax, -settings_.qmin));

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    if (contact_states[i]) {
      MultibodyWrenchConeResidual wrench_residual = MultibodyWrenchConeResidual(
          space.ndx(), handler_.get_rmodel(), actuation_matrix_, cms,
          prox_settings_, handler_.get_ee_name(i), settings_.mu,
          settings_.Lfoot, settings_.Wfoot);
      stm.addConstraint(wrench_residual, NegativeOrthant());
    }
  }

  return stm;
}

void FullDynamicsProblem::set_reference_poses(
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

void FullDynamicsProblem::set_reference_pose(const std::size_t t,
                                             const std::string &ee_name,
                                             const pinocchio::SE3 &pose_ref) {
  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
  cfr->setReference(pose_ref);
}

void FullDynamicsProblem::set_terminal_reference_pose(
    const std::string &ee_name, const pinocchio::SE3 &pose_ref) {
  CostStack *cs = get_terminal_cost_stack();
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[terminal_cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
  cfr->setReference(pose_ref);
}

void FullDynamicsProblem::set_reference_forces(
    const std::size_t t,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  CostStack *cs = get_cost_stack(t);
  if (force_refs.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "force_refs size does not match number of end effectors");
  }
  for (auto ee_name : handler_.get_ee_names()) {
    QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
        &*cs->components_[cost_map_.at(ee_name + "_force_cost")]);
    ContactForceResidual *cfr =
        dynamic_cast<ContactForceResidual *>(&*qrc->residual_);
    cfr->setReference(force_refs.at(ee_name));
  }
}

void FullDynamicsProblem::set_reference_force(
    const std::size_t i, const std::string &ee_name,
    const Eigen::VectorXd &force_ref) {
  CostStack *cs = get_cost_stack(i);
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_force_cost")]);
  ContactForceResidual *cfr =
      dynamic_cast<ContactForceResidual *>(&*qrc->residual_);
  cfr->setReference(force_ref);
}

pinocchio::SE3
FullDynamicsProblem::get_reference_pose(const std::size_t t,
                                        const std::string &ee_name) {
  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_pose_cost")]);
  FramePlacementResidual *cfr =
      dynamic_cast<FramePlacementResidual *>(&*qc->residual_);
  return cfr->getReference();
}

Eigen::VectorXd
FullDynamicsProblem::get_reference_force(const std::size_t t,
                                         const std::string &ee_name) {
  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_force_cost")]);
  ContactForceResidual *cfr =
      dynamic_cast<ContactForceResidual *>(&*qc->residual_);
  return cfr->getReference();
}

Eigen::VectorXd
FullDynamicsProblem::get_x0_from_multibody(const Eigen::VectorXd &x_multibody) {
  if (x_multibody.size() != handler_.get_x0().size()) {
    throw std::runtime_error("x_multibody is of incorrect size");
  }
  handler_.updateInternalData(x_multibody);
  return x_multibody;
}

CostStack FullDynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  auto cent_mom = CentroidalMomentumResidual(
      ter_space.ndx(), nu_, handler_.get_rmodel(), Eigen::VectorXd::Zero(6));

  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));
  term_cost.addCost(
      QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent));
  for (auto const &name : handler_.get_ee_names()) {
    FramePlacementResidual frame_residual = FramePlacementResidual(
        ter_space.ndx(), nu_, handler_.get_rmodel(), handler_.get_ee_pose(name),
        handler_.get_ee_id(name));

    term_cost.addCost(
        QuadraticResidualCost(ter_space, frame_residual, settings_.w_frame));
  }

  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ter_space.ndx(), nu_, handler_.get_rmodel(), handler_.get_com_position());

  StageConstraint term_constraint_com = {com_cstr, EqualityConstraint()};
  problem_->addTerminalConstraint(term_constraint_com);

  return term_cost;
}

void FullDynamicsProblem::updateTerminalConstraint() {
  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ndx_, nu_, handler_.get_rmodel(), handler_.get_com_position());

  StageConstraint term_constraint_com = {com_cstr, EqualityConstraint()};
  problem_->removeTerminalConstraints();
  problem_->addTerminalConstraint(term_constraint_com);
}

} // namespace simple_mpc
