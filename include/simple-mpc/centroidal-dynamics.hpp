///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
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
  /// @brief reference 0 state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  /// @brief timestep in problem shooting nodes
  double DT;

  Eigen::MatrixXd w_x;
  Eigen::MatrixXd w_u;

  Eigen::Vector3d gravity;
  int force_size;

  Eigen::Matrix3d w_linear_mom;
  Eigen::Matrix3d w_angular_mom;
  Eigen::Matrix3d w_linear_acc;
  Eigen::Matrix3d w_angular_acc;
};

class CentroidalProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CentroidalProblem();
  CentroidalProblem(const CentroidalSettings &settings,
                    const RobotHandler &handler);

  virtual ~CentroidalProblem(){};

  StageModel
  create_stage(const ContactMap &contact_map,
               const std::map<std::string, Eigen::VectorXd> &force_refs);

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &force_sequence);

  void
  set_reference_poses(const std::size_t t,
                      const std::map<std::string, pinocchio::SE3> &pose_refs) {}
  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) {
    return pinocchio::SE3::Identity();
  }
  Eigen::VectorXd get_x0_from_multibody(const Eigen::VectorXd &x_multibody);

  void compute_control_from_forces(
      const std::map<std::string, Eigen::VectorXd> &force_refs);
  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs);
  void set_reference_forces(const std::size_t t, const std::string &ee_name,
                            Eigen::VectorXd &force_ref);
  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name);
  CostStack create_terminal_cost();

protected:
  CentroidalSettings settings_;
  int nx_;
};

} // namespace simple_mpc
