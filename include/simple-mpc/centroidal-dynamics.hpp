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
#include <aligator/modelling/centroidal/linear-momentum.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>

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

/**
 * @brief Build a full dynamics problem
 */

struct CentroidalSettings {
  // reference state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  // timestep in problem shooting nodes
  double DT;

  // Cost function weights
  Eigen::MatrixXd w_x;           // State
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
  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;

  // Create one Centroidal terminal cost
  CostStack create_terminal_cost() override;

  // Getters and setters for pose not implemented
  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override {}
  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) override {
    return pinocchio::SE3::Identity();
  }

  // Getters and setters for contact forces
  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;
  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override;
  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) override;
  void compute_control_from_forces(
      const std::map<std::string, Eigen::VectorXd> &force_refs);

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override;
  CentroidalSettings get_settings() { return settings_; }

protected:
  CentroidalSettings settings_;
  int nx_;
};

} // namespace simple_mpc
