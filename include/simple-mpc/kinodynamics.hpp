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

struct KinodynamicsSettings : public Settings {
  Eigen::VectorXd w_cent;
  Eigen::Vector3d w_centder;

  KinodynamicsSettings();
  virtual ~KinodynamicsSettings() {}
};

class KinodynamicsProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KinodynamicsProblem();
  KinodynamicsProblem(const KinodynamicsSettings &settings,
                      const RobotHandler &handler);

  virtual ~KinodynamicsProblem(){};

  StageModel create_stage(const ContactMap &contact_map,
                          const std::vector<Eigen::VectorXd> &force_refs);
  void set_reference_poses(const std::size_t i,
                           const std::vector<pinocchio::SE3> &pose_refs);
  void set_reference_control(const std::size_t i, const Eigen::VectorXd &u_ref);
  void set_reference_forces(const std::size_t i,
                            const std::vector<Eigen::VectorXd> &force_refs);
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
