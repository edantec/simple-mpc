///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/modelling/centroidal/centroidal-wrench-cone.hpp>
#include <aligator/modelling/centroidal/friction-cone.hpp>
#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>
#include <aligator/modelling/multibody/center-of-mass-translation.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using KinodynamicsFwdDynamics = dynamics::KinodynamicsFwdDynamicsTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using CentroidalMomentumDerivativeResidual =
    CentroidalMomentumDerivativeResidualTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
using CenterOfMassTranslationResidual =
    CenterOfMassTranslationResidualTpl<double>;
using StateErrorResidual = StateErrorResidualTpl<double>;
using FunctionSliceXpr = FunctionSliceXprTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using CentroidalWrenchConeResidual = CentroidalWrenchConeResidualTpl<double>;
using FrictionConeResidual = FrictionConeResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
/**
 * @brief Build a kinodynamics problem based on
 * the KinodynamicsFwdDynamics object of Aligator.
 *
 * State is defined as concatenation of joint positions and
 * joint velocities; control is defined as concatenation of
 * contact forces and joint acceleration.
 */

struct KinodynamicsSettings {
  /// reference for state and control residuals
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;

  /// timestep in problem shooting nodes
  double DT;

  // Cost function weights
  Eigen::MatrixXd w_x;       // State
  Eigen::MatrixXd w_u;       // Control
  Eigen::MatrixXd w_frame;   // End effector placement
  Eigen::MatrixXd w_cent;    // Centroidal momentum
  Eigen::MatrixXd w_centder; // Derivative of centroidal momentum

  // Kinematics limits
  Eigen::VectorXd qmin;
  Eigen::VectorXd qmax;

  // Physics parameters
  Eigen::Vector3d gravity;
  double mu;
  double Lfoot;
  double Wfoot;
  int force_size;
};

class KinodynamicsProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Constructor
  KinodynamicsProblem();
  KinodynamicsProblem(const RobotHandler &handler);
  KinodynamicsProblem(const KinodynamicsSettings &settings,
                      const RobotHandler &handler);
  void initialize(const KinodynamicsSettings &settings);
  virtual ~KinodynamicsProblem() {};

  // Create one Kinodynamics stage
  StageModel createStage(
      const std::map<std::string, bool> &contact_phase,
      const std::map<std::string, pinocchio::SE3> &contact_pose,
      const std::map<std::string, Eigen::VectorXd> &contact_force) override;

  // Manage terminal cost and constraint
  CostStack createTerminalCost() override;
  void createTerminalConstraint() override;
  void updateTerminalConstraint(const Eigen::Vector3d &com_ref) override;

  // Getters and setters
  void setReferencePose(const std::size_t t, const std::string &ee_name,
                        const pinocchio::SE3 &pose_ref) override;
  void setReferencePoses(
      const std::size_t i,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override;
  void setTerminalReferencePose(const std::string &ee_name,
                                const pinocchio::SE3 &pose_ref) override;
  void setReferenceForces(
      const std::size_t i,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;
  void setReferenceForce(const std::size_t i, const std::string &ee_name,
                         const Eigen::VectorXd &force_ref) override;
  const Eigen::VectorXd
  getReferenceForce(const std::size_t i, const std::string &cost_name) override;
  const pinocchio::SE3 getReferencePose(const std::size_t i,
                                        const std::string &cost_name) override;
  const Eigen::VectorXd getProblemState() override;
  size_t getContactSupport(const std::size_t t) override;

  void computeControlFromForces(
      const std::map<std::string, Eigen::VectorXd> &force_refs);

  KinodynamicsSettings getSettings() { return settings_; }

protected:
  KinodynamicsSettings settings_;
};

} // namespace simple_mpc
