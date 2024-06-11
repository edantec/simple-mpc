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
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/contact-force.hpp>
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
using StageModel = StageModelTpl<double>;
using CostStack = CostStackTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using MultibodyConstraintFwdDynamics =
    dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using ODEAbstract = dynamics::ODEAbstractTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using ContactMap = ContactMapTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using TrajOptProblem = TrajOptProblemTpl<double>;
using ContactForceResidual = ContactForceResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using ControlErrorResidual = ControlErrorResidualTpl<double>;
using StateErrorResidual = StateErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct FullDynamicsSettings : public Settings {
public:
  Eigen::VectorXd w_forces;
  Eigen::VectorXd w_frame;

  Eigen::VectorXd umin;
  Eigen::VectorXd umax;

  Eigen::VectorXd qmin;
  Eigen::VectorXd qmax;
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

  StageModel create_stage(const ContactMap &contact_map,
                          const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_poses(const std::size_t i,
                           const std::vector<pinocchio::SE3> &pose_refs);
  void set_reference_forces(const std::size_t i,
                            const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_forces(const std::size_t i, const std::string &ee_name,
                            Eigen::VectorXd &force_ref);
  CostStack create_terminal_cost();

protected:
  FullDynamicsSettings settings_;
  Eigen::MatrixXd actuation_matrix_;
  ProximalSettings prox_settings_;
  pinocchio::context::RigidConstraintModelVector constraint_models_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
