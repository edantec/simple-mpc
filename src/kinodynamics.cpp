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
    : Base(handler), settings_(settings) {

  control_ref_ = settings_.u0;

  // Set up cost names used in kinodynamics problem
  int cost_incr = 0;
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
    const std::vector<Eigen::VectorXd> &force_refs) {
  auto space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  for (std::size_t i = 0; i < force_refs.size(); i++) {
    control_ref_.segment((long)i * force_refs[0].size(), force_refs[0].size()) =
        force_refs[i];
  }

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
    FramePlacementResidual frame_residual =
        FramePlacementResidual(space.ndx(), nu_, handler_.get_rmodel(),
                               frame_placement, handler_.get_ee_id(i));
    int is_active = 0;
    if (!contact_states[i])
      is_active = 1;

    rcost.addCost(QuadraticResidualCost(space, frame_residual,
                                        settings_.w_frame * is_active));
  }

  KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, handler_.get_rmodel(), settings_.gravity, contact_states,
      handler_.get_ee_ids(), (int)force_refs[0].size());
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

void KinodynamicsProblem::set_reference_control(const std::size_t i,
                                                const Eigen::VectorXd &u_ref) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);
  QuadraticControlCost *qc = dynamic_cast<QuadraticControlCost *>(
      &*cs->components_[cost_map_.at("control_cost")]);

  qc->setTarget(u_ref);
}

CostStack KinodynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(handler_.get_rmodel());
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

} // namespace simple_mpc
