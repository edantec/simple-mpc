///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/core/cost-abstract.hpp>
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
using CostAbstract = CostAbstractTpl<double>;

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
  Problem(const RobotHandler &handler);
  virtual ~Problem() {}

  virtual StageModel
  create_stage(const ContactMap &contact_map,
               const std::vector<Eigen::VectorXd> &force_refs);
  virtual CostStack create_terminal_cost();
  virtual void create_problem(const Eigen::VectorXd &x0,
                              const std::vector<ContactMap> &contact_sequence);

  virtual void
  set_reference_poses(const std::size_t i,
                      const std::vector<pinocchio::SE3> &pose_refs);
  virtual void
  set_reference_forces(const std::size_t i,
                       const std::vector<Eigen::VectorXd> &force_refs);
  virtual void set_reference_control(const std::size_t i,
                                     const Eigen::VectorXd &u_ref);
  virtual void insert_cost(CostStack &cost_stack,
                           const xyz::polymorphic<CostAbstract> &cost,
                           std::map<std::string, std::size_t> &cost_map,
                           const std::string &name, int &cost_incr);

  /// @brief The reference shooting problem storing all shooting nodes
  std::shared_ptr<aligator::context::TrajOptProblem> problem_;

  /// @brief The robot model
  RobotHandler handler_;

  /// @brief Dictionnary of cost name + cost id
  std::map<std::string, std::size_t> cost_map_;

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
