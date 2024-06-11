///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

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

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings settings,
                                         const RobotHandler &handler)
    : Base(handler), settings_(settings) {
  actuation_matrix_.resize(nv_, nu_);
  actuation_matrix_.setZero();
  actuation_matrix_.bottomRows(nu_).setIdentity();

  prox_settings_ = ProximalSettings(1e-9, 1e-10, 10);

  for (std::size_t i = 0; i < handler_.get_ee_ids().size(); i++) {
    auto frame_ids = handler_.get_ee_id(i);
    auto joint_ids = handler_.get_rmodel().frames[frame_ids].parentJoint;
    pinocchio::SE3 pl1 = handler_.get_rmodel().frames[frame_ids].placement;
    pinocchio::SE3 pl2 = handler_.get_ee_frame(i);
    pinocchio::RigidConstraintModel constraint_model =
        pinocchio::RigidConstraintModel(pinocchio::ContactType::CONTACT_6D,
                                        handler_.get_rmodel(), joint_ids, pl1,
                                        0, pl2, pinocchio::LOCAL_WORLD_ALIGNED);
    constraint_model.corrector.Kp << 0, 0, 100, 0, 0, 0;
    constraint_model.corrector.Kd << 50, 50, 50, 50, 50, 50;
    constraint_model.name = handler_.get_ee_name(i);
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
  for (auto cname : handler_.get_ee_names()) {
    cost_map_.insert({cname + "_pose_cost", cost_incr});
    cost_incr++;
  }
  for (auto cname : handler_.get_ee_names()) {
    cost_map_.insert({cname + "_force_cost", cost_incr});
    cost_incr++;
  }
}

StageModel FullDynamicsProblem::create_stage(
    const ContactMap &contact_map,
    const std::vector<Eigen::VectorXd> &force_refs) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, settings_.u0, settings_.w_u));

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.get_rmodel(), Eigen::VectorXd::Zero(6));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));

  pinocchio::context::RigidConstraintModelVector cms;
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  if (contact_states.size() != handler_.get_ee_ids().size()) {
    throw std::runtime_error(
        "contact states size does not match number of end effectors");
  }

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
    frame_placement.translation() = contact_poses[i];
    FramePlacementResidual frame_residual =
        FramePlacementResidual(space.ndx(), nu_, handler_.get_rmodel(),
                               frame_placement, handler_.get_ee_id(i));

    int is_active = 0;
    if (contact_states[i])
      cms.push_back(constraint_models_[i]);
    else
      is_active = 1;

    rcost.addCost(QuadraticResidualCost(space, frame_residual,
                                        settings_.w_frame * is_active));
  }

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    ContactForceResidual frame_force = ContactForceResidual(
        space.ndx(), handler_.get_rmodel(), actuation_matrix_, cms,
        prox_settings_, force_refs[i], handler_.get_ee_name(i));
    int is_active = 0;
    if (contact_states[i])
      is_active = 1;
    rcost.addCost(QuadraticResidualCost(space, frame_force,
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
  stm.addConstraint(state_fn, BoxConstraint(-settings_.qmax, -settings_.qmin));

  return stm;
}

void FullDynamicsProblem::set_reference_poses(
    const std::size_t i, const std::vector<pinocchio::SE3> &pose_refs) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  if (pose_refs.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }

  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);
  for (std::size_t i = 0; i < pose_refs.size(); i++) {
    QuadraticResidualCost *qrc =
        dynamic_cast<QuadraticResidualCost *>(&*cs->components_[cost_map_.at(
            handler_.get_ee_names()[i] + "_pose_cost")]);
    FramePlacementResidual *cfr =
        dynamic_cast<FramePlacementResidual *>(&*qrc->residual_);
    cfr->setReference(pose_refs[i]);
  }
}

void FullDynamicsProblem::set_reference_forces(
    const std::size_t i, const std::vector<Eigen::VectorXd> &force_refs) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  if (force_refs.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "force_refs size does not match number of end effectors");
  }

  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);
  for (std::size_t i = 0; i < force_refs.size(); i++) {
    QuadraticResidualCost *qrc =
        dynamic_cast<QuadraticResidualCost *>(&*cs->components_[cost_map_.at(
            handler_.get_ee_names()[i] + "_force_cost")]);
    ContactForceResidual *cfr =
        dynamic_cast<ContactForceResidual *>(&*qrc->residual_);
    cfr->setReference(force_refs[i]);
  }
}

void FullDynamicsProblem::set_reference_forces(const std::size_t i,
                                               const std::string &ee_name,
                                               Eigen::VectorXd &force_ref) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }

  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);
  QuadraticResidualCost *qrc = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at(ee_name + "_force_cost")]);
  ContactForceResidual *cfr =
      dynamic_cast<ContactForceResidual *>(&*qrc->residual_);
  cfr->setReference(force_ref);
}

CostStack FullDynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

} // namespace simple_mpc
