///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

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
  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;

  // Create one Kinodynamics terminal cost
  CostStack create_terminal_cost() override;

  // Getters and setters
  void set_reference_poses(
      const std::size_t i,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override;
  void set_reference_forces(
      const std::size_t i,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override;
  void set_reference_force(const std::size_t i, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override;
  Eigen::VectorXd get_reference_force(const std::size_t i,
                                      const std::string &cost_name) override;
  pinocchio::SE3 get_reference_pose(const std::size_t i,
                                    const std::string &cost_name) override;
  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override;
  void compute_control_from_forces(
      const std::map<std::string, Eigen::VectorXd> &force_refs);

  KinodynamicsSettings get_settings() { return settings_; }

protected:
  KinodynamicsSettings settings_;
};

} // namespace simple_mpc
