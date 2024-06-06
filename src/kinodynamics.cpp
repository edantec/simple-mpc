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

#include "simple-mpc/kinodynamics.hpp"

namespace simple_mpc {
using namespace aligator;

KinodynamicsProblem::KinodynamicsProblem(const KinodynamicsSettings &settings,
                                         const pinocchio::Model &rmodel) {
  initialize(settings, rmodel);
}

void KinodynamicsProblem::initialize(const KinodynamicsSettings &settings,
                                     const pinocchio::Model &rmodel) {
  settings_ = settings;
  rmodel_ = rmodel;
  rdata_ = pinocchio::Data(rmodel_);

  nq_ = rmodel.nq;
  nv_ = rmodel.nv;
  nu_ = nv_ - 6;

  actuation_matrix_.resize(nv_, nu_);
  actuation_matrix_.setZero();
  actuation_matrix_.bottomRows(nu_).setIdentity();

  pinocchio::forwardKinematics(rmodel_, rdata_, settings_.x0.head(nq_));
  pinocchio::updateFramePlacements(rmodel, rdata_);

  prox_settings_ = ProximalSettings(1e-9, 1e-10, 10);
  
  for (auto &ee_name : settings_.end_effectors) {
    frame_ids_vector_.push_back(rmodel_.getFrameId(ee_name));
  }
}

StageModel KinodynamicsProblem::create_stage(ContactMap &contact_map) {
  auto space = MultibodyPhaseSpace(rmodel_);
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  auto cent_mom = CentroidalMomentumResidual(
        space.ndx(), nu_, rmodel_, Eigen::VectorXd::Zero(6)
  );
  auto centder_mom = CentroidalMomentumDerivativeResidual(
        space.ndx(), rmodel_, settings_.gravity, contact_states, frame_ids_vector_, 6
  );

  rcost.addCost(QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost(QuadraticControlCost(space, settings_.u0, settings_.w_u));
  rcost.addCost(QuadraticResidualCost(space, cent_mom, settings_.w_cent));
  rcost.addCost(QuadraticResidualCost(space, centder_mom, settings_.w_centder));

  for (std::size_t i = 0; i < contact_states.size(); i++) {
    if (not(contact_states[i])) {
      pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
      frame_placement.translation() = contact_poses[i];
      FramePlacementResidual frame_residual = FramePlacementResidual(
          space.ndx(), nu_, rmodel_, frame_placement,
          rmodel_.getFrameId(settings_.end_effectors[i]));

      rcost.addCost(
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
  }

  KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, rmodel_, settings_.gravity, contact_states, frame_ids_vector_, 6);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

CostStack KinodynamicsProblem::create_terminal_cost() {
  auto ter_space = MultibodyPhaseSpace(rmodel_);
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

void KinodynamicsProblem::create_problem(
    std::vector<ContactMap> contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    stage_models.push_back(create_stage(cm));
  }

  problem_ = std::make_shared<TrajOptProblem>(settings_.x0, stage_models,
                                              create_terminal_cost());
}

} // namespace simple_mpc
