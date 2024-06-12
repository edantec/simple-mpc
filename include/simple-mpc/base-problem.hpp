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

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/contact-map.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>
#include <aligator/modelling/multibody/contact-force.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
using namespace aligator;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using ContactMap = aligator::ContactMapTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;
using CostAbstract = CostAbstractTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using VectorSpace = proxsuite::nlp::VectorSpaceTpl<double>;
using CentroidalFwdDynamics = dynamics::CentroidalFwdDynamicsTpl<double>;
using ContactForceResidual = ContactForceResidualTpl<double>;
/**
 * @brief Build a full dynamics problem
 */

class Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Problem();
  Problem(const RobotHandler &handler);
  virtual ~Problem();

  virtual StageModel
  create_stage(const ContactMap &contact_map,
               const std::vector<Eigen::VectorXd> &force_refs) = 0;
  virtual CostStack create_terminal_cost() = 0;
  virtual void
  create_problem(const Eigen::VectorXd &x0,
                 const std::vector<ContactMap> &contact_sequence) = 0;
  std::vector<xyz::polymorphic<StageModel>>
  create_stages(const std::vector<ContactMap> &contact_sequence);

  virtual void
  set_reference_poses(const std::size_t i,
                      const std::vector<pinocchio::SE3> &pose_refs) = 0;
  virtual pinocchio::SE3 get_reference_pose(const std::size_t i,
                                            const std::string &ee_name) = 0;

  virtual void
  set_reference_forces(const std::size_t i,
                       const std::vector<Eigen::VectorXd> &force_refs) = 0;
  virtual void set_reference_forces(const std::size_t i,
                                    const std::string &ee_name,
                                    Eigen::VectorXd &force_ref) = 0;
  void set_reference_control(const std::size_t i, const Eigen::VectorXd &u_ref);
  Eigen::VectorXd get_reference_control(const std::size_t i);
  virtual Eigen::VectorXd get_reference_force(const std::size_t i,
                                              const std::string &ee_name) = 0;
  void insert_cost(CostStack &cost_stack,
                   const xyz::polymorphic<CostAbstract> &cost,
                   std::map<std::string, std::size_t> &cost_map,
                   const std::string &name, int &cost_incr);

  CostStack *get_cost_stack(std::size_t i);
  std::size_t get_cost_number();
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
