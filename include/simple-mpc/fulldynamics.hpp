///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_HPP_
#define SIMPLE_MPC_HPP_

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <pinocchio/algorithm/proximal.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using IntegratorSemiImplEuler =
    aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using MultibodyConstraintFwdDynamics =
    aligator::dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using ContactMap = aligator::ContactMapTpl<double>;
using FramePlacementResidual = aligator::FramePlacementResidualTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

 struct FullDynamicsSettings {
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

    Eigen::VectorXd w_x;
    Eigen::VectorXd w_u;
    Eigen::VectorXd w_frame;
};

class FullDynamicsProblem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FullDynamicsProblem();
  FullDynamicsProblem(const FullDynamicsSettings settings,
                      const pinocchio::Model &rmodel);
  void initialize(const FullDynamicsSettings settings,
                  const pinocchio::Model &rmodel);
  virtual ~FullDynamicsProblem() {}

  StageModel create_stage(ContactMap &contact_map);
  CostStack create_terminal_cost();
  void create_problem(std::vector<ContactMap> contact_sequence);

  /// @brief Parameters to tune the algorithm, given at init.
  FullDynamicsSettings settings_;

  /// @brief The reference shooting problem storing all shooting nodes
  std::shared_ptr<aligator::context::TrajOptProblem> problem_;

  /// @brief The robot model
  pinocchio::Model rmodel_;

  /// @brief Robot data
  pinocchio::Data rdata_;

  /// @brief List of stage models forming the horizon
  std::vector<xyz::polymorphic<StageModel>> stage_models_;

protected:
  Eigen::MatrixXd actuation_matrix_;
  int nq_;
  int nv_;
  int nu_;
  ProximalSettings prox_settings_;
  pinocchio::context::RigidConstraintModelVector constraint_models_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
