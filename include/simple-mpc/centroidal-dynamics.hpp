///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef SIMPLE_MPC_CENTROIDAL_DYNAMICS_HPP_
#define SIMPLE_MPC_CENTROIDAL_DYNAMICS_HPP_

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

struct CentroidalSettings : public Settings {
  Eigen::VectorXd w_linear_mom;
  Eigen::Vector3d w_angular_mom;
  Eigen::VectorXd w_linear_acc;
  Eigen::Vector3d w_angular_acc;

  CentroidalSettings();
  virtual ~CentroidalSettings() {}
};

class CentroidalProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CentroidalProblem();
  CentroidalProblem(const CentroidalSettings &settings,
                    const RobotHandler &handler);

  virtual ~CentroidalProblem(){};

  StageModel create_stage(const ContactMap &contact_map,
                          const std::vector<Eigen::VectorXd> &force_refs);
  void
  compute_control_from_forces(const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_poses(const std::size_t i,
                           const std::vector<pinocchio::SE3> &pose_refs) {};
  void set_reference_forces(const std::size_t i,
                            const std::vector<Eigen::VectorXd> &force_refs);
  CostStack create_terminal_cost();

  Eigen::VectorXd control_ref_;

protected:
  CentroidalSettings settings_;
  int nx_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_CENTROIDAL_DYNAMICS_HPP_
