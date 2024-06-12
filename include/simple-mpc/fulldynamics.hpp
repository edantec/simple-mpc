///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_FULLDYNAMICS_HPP_
#define SIMPLE_MPC_FULLDYNAMICS_HPP_

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
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
using MultibodyConstraintFwdDynamics =
    dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using ControlErrorResidual = ControlErrorResidualTpl<double>;
using StateErrorResidual = StateErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct FullDynamicsSettings {
public:
  /// @brief reference 0 state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  /// @brief timestep in problem shooting nodes
  double DT;

  Eigen::MatrixXd w_x;
  Eigen::MatrixXd w_u;
  Eigen::MatrixXd w_cent;
  Eigen::MatrixXd w_centder;

  Eigen::Vector3d gravity;
  int force_size;

  Eigen::MatrixXd w_forces;
  Eigen::MatrixXd w_frame;

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
  pinocchio::SE3 get_reference_pose(const std::size_t i,
                                    const std::string &cost_name);
  Eigen::VectorXd get_reference_force(const std::size_t i,
                                      const std::string &cost_name);
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
