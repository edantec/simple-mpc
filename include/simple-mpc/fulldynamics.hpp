///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <aligator/modelling/function-xpr-slice.hpp>
#include <aligator/modelling/multibody/center-of-mass-translation.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/multibody-wrench-cone.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using Base = Problem;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using MultibodyConstraintFwdDynamics =
    dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using ControlErrorResidual = ControlErrorResidualTpl<double>;
using StateErrorResidual = StateErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using StageConstraint = StageConstraintTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
using MultibodyWrenchConeResidual =
    aligator::MultibodyWrenchConeResidualTpl<double>;
using CenterOfMassTranslationResidual =
    CenterOfMassTranslationResidualTpl<double>;
using FunctionSliceXpr = FunctionSliceXprTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct FullDynamicsSettings {
public:
  // reference 0 state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;

  // timestep in problem shooting nodes
  double DT;

  // Cost function weights
  Eigen::MatrixXd w_x;      // State
  Eigen::MatrixXd w_u;      // Control
  Eigen::MatrixXd w_cent;   // Centroidal momentum
  Eigen::MatrixXd w_forces; // Contact forces
  Eigen::MatrixXd w_frame;  // End effector placement

  // Physics parameters
  Eigen::Vector3d gravity;
  int force_size;
  double mu;
  double Lfoot; // Half-length of foot (if contact 6D)
  double Wfoot; // Half-width of foot (if contact 6D)

  // Control limits
  Eigen::VectorXd umin;
  Eigen::VectorXd umax;

  // Kinematics limits
  Eigen::VectorXd qmin;
  Eigen::VectorXd qmax;
};

class FullDynamicsProblem : public Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Constructors
  FullDynamicsProblem();
  FullDynamicsProblem(const RobotHandler &handler);
  FullDynamicsProblem(const FullDynamicsSettings &settings,
                      const RobotHandler &handler);
  void initialize(const FullDynamicsSettings &settings);
  virtual ~FullDynamicsProblem() {}

  // Create one FullDynamics stage
  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;

  // Create one FullDynamics terminal cost
  CostStack create_terminal_cost() override;

  void updateTerminalConstraint() override;

  // Getters and setters
  void set_reference_pose(const std::size_t t, const std::string &ee_name,
                          const pinocchio::SE3 &pose_ref) override;
  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override;
  void set_terminal_reference_pose(const std::string &ee_name,
                                   const pinocchio::SE3 &pose_ref) override;
  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;
  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override;
  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &cost_name) override;
  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &cost_name) override;
  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override;
  FullDynamicsSettings get_settings() { return settings_; }

protected:
  // Problem settings
  FullDynamicsSettings settings_;
  ProximalSettings prox_settings_;

  // Actuation matrix
  Eigen::MatrixXd actuation_matrix_;

  // Complete list of contact models
  pinocchio::context::RigidConstraintModelVector constraint_models_;
};

} // namespace simple_mpc
