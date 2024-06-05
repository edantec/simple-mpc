///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_HPP_
#define SIMPLE_MPC_HPP_

#include <Eigen/src/Core/Matrix.h>
#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
/**
 * @brief Build a full dynamics problem
 */

struct FullDynamicsSettings {
  /// @brief reference 0 state
  Eigen::VectorXd x0;
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

  FullDynamicsSettings();
  virtual ~FullDynamicsSettings() {}
};

class FullDynamicsProblem {
  typedef std::vector<aligator::context::StageModel> StageList;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FullDynamicsProblem();
  FullDynamicsProblem(const FullDynamicsSettings &settings,
                      const pinocchio::Model &rmodel);
  void initialize(const FullDynamicsSettings &settings,
                  const pinocchio::Model &rmodel);
  virtual ~FullDynamicsProblem() {}

  /// @brief Parameters to tune the algorithm, given at init.
  FullDynamicsSettings settings_;

  /// @brief The reference shooting problem storing all shooting nodes
  std::shared_ptr<aligator::context::TrajOptProblem> problem_;

  /// @brief The manifold space for multibody dynamics
  std::shared_ptr<MultibodyPhaseSpace> space_;

  /// @brief The robot model
  pinocchio::Model rmodel_;

protected:
  double reg;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
