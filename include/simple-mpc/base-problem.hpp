///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef SIMPLE_MPC_BASEDYNAMICS_HPP_
#define SIMPLE_MPC_BASEDYNAMICS_HPP_

#include <aligator/modelling/contact-map.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
using namespace aligator;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using ContactMap = aligator::ContactMapTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct Settings {
  /// @brief reference 0 state and control
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  /// @brief Duration of the OCP horizon.
  int T;
  /// @brief timestep in problem shooting nodes
  double DT;
  /// @brief stop threshold to configure the solver
  double solver_th_stop;
  /// @brief solver param reg_min
  double solver_reg_min;
  /// @brief Solver max number of iteration
  int solver_maxiter;
  /// @brief List of end effector names
  std::vector<std::string> end_effectors;
  /// @brief List of controlled joint names
  std::vector<std::string> controlled_joints_names;

  Eigen::MatrixXd w_x;
  Eigen::MatrixXd w_u;
  Eigen::MatrixXd w_frame;
  Eigen::MatrixXd w_cent;
  Eigen::MatrixXd w_centder;

  Eigen::Vector3d gravity;

  Settings();
  virtual ~Settings() {}
};

class Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Problem();
  Problem(const Settings &settings, const RobotHandler &handler);
  void initialize(const Settings &settings, const RobotHandler &handler);
  virtual ~Problem() {}

  virtual StageModel create_stage(ContactMap &contact_map);
  virtual CostStack create_terminal_cost();
  virtual void create_problem(std::vector<ContactMap> contact_sequence);

  /// @brief Parameters to tune the algorithm, given at init.
  Settings settings_;

  /// @brief The reference shooting problem storing all shooting nodes
  std::shared_ptr<aligator::context::TrajOptProblem> problem_;

  /// @brief The robot model
  RobotHandler handler_;

  /// @brief List of stage models forming the horizon
  std::vector<xyz::polymorphic<StageModel>> stage_models_;

protected:
  int nq_;
  int nv_;
  int nu_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
