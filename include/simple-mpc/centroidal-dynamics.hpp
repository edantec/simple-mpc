///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/modelling/centroidal/angular-acceleration.hpp>
#include <aligator/modelling/centroidal/angular-momentum.hpp>
#include <aligator/modelling/centroidal/centroidal-acceleration.hpp>
#include <aligator/modelling/centroidal/centroidal-translation.hxx>
#include <aligator/modelling/centroidal/linear-momentum.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>
#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using VectorSpace = proxsuite::nlp::VectorSpaceTpl<double>;
using CentroidalFwdDynamics = dynamics::CentroidalFwdDynamicsTpl<double>;
using CentroidalAccelerationResidual =
    CentroidalAccelerationResidualTpl<double>;
using AngularAccelerationResidual = AngularAccelerationResidualTpl<double>;
using LinearMomentumResidual = LinearMomentumResidualTpl<double>;
using AngularMomentumResidual = AngularMomentumResidualTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
using StageConstraint = StageConstraintTpl<double>;
using CentroidalCoMResidual = CentroidalCoMResidualTpl<double>;
using ContactMap = ContactMapTpl<double>;

/**
 * @brief Build a centroidal dynamics problem based on
 * the CentroidalFwdDynamics from Aligator.
 *
 * State is defined as concatenation of center of mass position and
 * centroidal momentum; control is defined as concatenation of
 * contact forces.
 */

struct CentroidalSettings {
  // reference state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  // timestep in problem shooting nodes
  double DT;

  // Cost function weights
  Eigen::MatrixXd w_u;           // Control
  Eigen::Matrix3d w_linear_mom;  // Linear momentum
  Eigen::Matrix3d w_angular_mom; // Angular momentum
  Eigen::Matrix3d w_linear_acc;  // Linear acceleration
  Eigen::Matrix3d w_angular_acc; // Angular acceleration

  // Physics parameters
  Eigen::Vector3d gravity;
  int force_size;
};

class CentroidalProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Constructor
  CentroidalProblem();
  CentroidalProblem(const RobotHandler &handler);
  CentroidalProblem(const CentroidalSettings &settings,
                    const RobotHandler &handler);
  void initialize(const CentroidalSettings &settings);
  virtual ~CentroidalProblem() {};

  // Create one Centroidal stage
  StageModel createStage(
      const std::map<std::string, bool> &contact_phase,
      const std::map<std::string, pinocchio::SE3> &contact_pose,
      const std::map<std::string, Eigen::VectorXd> &contact_force) override;

  // Manage terminal cost and constraint
  CostStack createTerminalCost() override;
  void createTerminalConstraint() override;
  void updateTerminalConstraint(const Eigen::Vector3d &com_ref) override;

  // Getters and setters for pose not implemented
  void setReferencePose(const std::size_t t, const std::string &ee_name,
                        const pinocchio::SE3 &pose_ref) override;
  void setReferencePoses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override;
  void setTerminalReferencePose(const std::string &ee_name,
                                const pinocchio::SE3 &pose_ref) override {}
  const pinocchio::SE3 getReferencePose(const std::size_t t,
                                        const std::string &ee_name) override;

  // Getters and setters for contact forces
  void setReferenceForces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;
  void setReferenceForce(const std::size_t t, const std::string &ee_name,
                         const Eigen::VectorXd &force_ref) override;
  const Eigen::VectorXd getReferenceForce(const std::size_t t,
                                          const std::string &ee_name) override;
  void computeControlFromForces(
      const std::map<std::string, Eigen::VectorXd> &force_refs);

  const Eigen::VectorXd getProblemState() override;
  CentroidalSettings getSettings() { return settings_; }

protected:
  CentroidalSettings settings_;
  int nx_;
};

} // namespace simple_mpc
