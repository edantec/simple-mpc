///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_HPP_
#define SIMPLE_MPC_HPP_

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;

/**
 * @brief Generic settings form
 */

struct Settings {
  /// @brief reference 0 state and control
public:
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

  Settings();
  virtual ~Settings() = default;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
