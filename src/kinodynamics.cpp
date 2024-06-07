///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/kinodynamics.hpp"

namespace simple_mpc {
using namespace aligator;

KinodynamicsProblem::KinodynamicsProblem(const KinodynamicsSettings &settings,
                                         const RobotHandler &handler)
    : Base(settings, handler) {}

StageModel KinodynamicsProblem::create_stage(ContactMap &contact_map) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.get_rmodel(), Eigen::VectorXd::Zero(6));
  auto centder_mom = CentroidalMomentumDerivativeResidual(
      space.ndx(), handler_.get_rmodel(), settings_.gravity, contact_states,
      handler_.get_ee_ids(), 6);

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, settings_.u0, settings_.w_u));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));
  rcost.addCost(QuadraticResidualCost(space, centder_mom, settings_.w_centder));

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    if (not(contact_states[i])) {
      pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
      frame_placement.translation() = contact_poses[i];
      FramePlacementResidual frame_residual =
          FramePlacementResidual(space.ndx(), nu_, handler_.get_rmodel(),
                                 frame_placement, handler_.get_ee_id(i));

      rcost.addCost(
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
  }

  KinodynamicsFwdDynamics ode =
      KinodynamicsFwdDynamics(space, handler_.get_rmodel(), settings_.gravity,
                              contact_states, handler_.get_ee_ids(), 6);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

CostStack KinodynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

} // namespace simple_mpc
