///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef SIMPLE_MPC_KINODYNAMICS_HPP_
#define SIMPLE_MPC_KINODYNAMICS_HPP_

#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using KinodynamicsFwdDynamics = dynamics::KinodynamicsFwdDynamicsTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using CentroidalMomentumDerivativeResidual =
    CentroidalMomentumDerivativeResidualTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct KinodynamicsSettings {
  /// @brief reference 0 state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  /// @brief timestep in problem shooting nodes
  double DT;

  Eigen::MatrixXd w_x;
  Eigen::MatrixXd w_u;
  Eigen::MatrixXd w_frame;
  Eigen::MatrixXd w_cent;
  Eigen::MatrixXd w_centder;

  Eigen::VectorXd qmin;
  Eigen::VectorXd qmax;

  Eigen::Vector3d gravity;
  int force_size;
};

class KinodynamicsProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KinodynamicsProblem();
  KinodynamicsProblem(const KinodynamicsSettings &settings,
                      const RobotHandler &handler);

  virtual ~KinodynamicsProblem(){};

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence);

  StageModel create_stage(const ContactMap &contact_map,
                          const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_poses(const std::size_t i,
                           const std::vector<pinocchio::SE3> &pose_refs);
  void set_reference_forces(const std::size_t i,
                            const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_forces(const std::size_t i, const std::string &ee_name,
                            Eigen::VectorXd &force_ref);
  Eigen::VectorXd get_reference_force(const std::size_t i,
                                      const std::string &cost_name);
  pinocchio::SE3 get_reference_pose(const std::size_t i,
                                    const std::string &cost_name);
  void
  compute_control_from_forces(const std::vector<Eigen::VectorXd> &force_refs);
  CostStack create_terminal_cost();

  Eigen::VectorXd control_ref_;

  /// @brief Parameters to tune the algorithm, given at init.
  // KinodynamicsSettings settings_;

protected:
  KinodynamicsSettings settings_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
