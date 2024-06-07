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
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/fulldynamics.hpp"

namespace simple_mpc {
using namespace aligator;

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings settings,
                                         const RobotHandler &handler)
    : Base(settings, handler) {
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
}

StageModel FullDynamicsProblem::create_stage(ContactMap &contact_map) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, settings_.u0, settings_.w_u));

  pinocchio::context::RigidConstraintModelVector cms;
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();
  for (std::size_t i = 0; i < contact_states.size(); i++) {
    if (contact_states[i]) {
      cms.push_back(constraint_models_[i]);
    } else {
      pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
      frame_placement.translation() = contact_poses[i];
      FramePlacementResidual frame_residual =
          FramePlacementResidual(space.ndx(), nu_, handler_.get_rmodel(),
                                 frame_placement, handler_.get_ee_id(i));

      rcost.addCost(
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
  }

  MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(
      space, actuation_matrix_, cms, prox_settings_);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

CostStack FullDynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

void FullDynamicsProblem::create_problem(
    std::vector<ContactMap> contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    stage_models.push_back(create_stage(cm));
  }

  problem_ = std::make_shared<TrajOptProblem>(settings_.x0, stage_models,
                                              create_terminal_cost());
}

} // namespace simple_mpc
