///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_FULLDYNAMICS_HPP_
#define SIMPLE_MPC_FULLDYNAMICS_HPP_

#include "aligator/modelling/contact-map.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using Base = Problem;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using IntegratorSemiImplEuler =
    aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using MultibodyConstraintFwdDynamics =
    aligator::dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using ContactMap = aligator::ContactMapTpl<double>;
using FramePlacementResidual = aligator::FramePlacementResidualTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct FullDynamicsSettings : public Settings {
public:
  Eigen::VectorXd w_forces;
  Eigen::VectorXd w_frame;
};

class FullDynamicsProblem : public Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FullDynamicsProblem();
  FullDynamicsProblem(const FullDynamicsSettings settings,
                      const RobotHandler &handler);
  void initialize(const FullDynamicsSettings settings,
                  const RobotHandler &handler);
  virtual ~FullDynamicsProblem() {}

  StageModel create_stage(ContactMap &contact_map);
  CostStack create_terminal_cost();
  void create_problem(std::vector<ContactMap> contact_sequence);

protected:
  Eigen::MatrixXd actuation_matrix_;
  ProximalSettings prox_settings_;
  pinocchio::context::RigidConstraintModelVector constraint_models_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
