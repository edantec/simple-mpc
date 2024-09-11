///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/contact-map.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#ifndef ALIGATOR_PINOCCHIO_V3
#error "aligator no compile with pin v3"
#endif
#include <aligator/modelling/multibody/contact-force.hpp>

namespace simple_mpc {
using namespace aligator;
using StageModel = StageModelTpl<double>;
using CostStack = CostStackTpl<double>;
using ContactMap = ContactMapTpl<double>;
using TrajOptProblem = TrajOptProblemTpl<double>;
using CostAbstract = CostAbstractTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using VectorSpace = proxsuite::nlp::VectorSpaceTpl<double>;
using CentroidalFwdDynamics = dynamics::CentroidalFwdDynamicsTpl<double>;
using ContactForceResidual = ContactForceResidualTpl<double>;
/**
 * @brief Base problem abstract class
 */

class Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Constructor
  Problem();
  Problem(const RobotHandler &handler);
  virtual ~Problem();

  /// Virtual functions defined in child classes

  // Create one instance of stage from desired contacts and forces
  virtual StageModel
  create_stage(const ContactMap &contact_map,
               const std::map<std::string, Eigen::VectorXd> &force_refs) = 0;
  virtual CostStack create_terminal_cost() = 0;

  virtual void updateTerminalConstraint() = 0;

  // Setter and getter for poses reference
  virtual void set_reference_pose(const std::size_t t,
                                  const std::string &ee_name,
                                  const pinocchio::SE3 &pose_ref) = 0;
  virtual void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) = 0;
  virtual void set_terminal_reference_pose(const std::string &ee_name,
                                           const pinocchio::SE3 &pose_ref) = 0;
  virtual pinocchio::SE3 get_reference_pose(const std::size_t t,
                                            const std::string &ee_name) = 0;

  // Setter and getter for forces reference
  virtual void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) = 0;
  virtual void set_reference_force(const std::size_t t,
                                   const std::string &ee_name,
                                   const Eigen::VectorXd &force_ref) = 0;
  virtual Eigen::VectorXd get_reference_force(const std::size_t t,
                                              const std::string &ee_name) = 0;
  virtual Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) = 0;
  /// Common functions to all types of problems

  // Create the complete vector of stages from contact_sequence
  virtual std::vector<xyz::polymorphic<StageModel>>
  create_stages(const std::vector<ContactMap> &contact_sequence,
                const std::vector<std::map<std::string, Eigen::VectorXd>>
                    &force_sequence);

  // Create one TrajOptProblem from contact sequence
  void create_problem(const Eigen::VectorXd &x0, const size_t horizon,
                      const int force_size, const double gravity);

  // Setter and getter for control reference
  void set_reference_control(const std::size_t t, const Eigen::VectorXd &u_ref);

  Eigen::VectorXd get_reference_control(const std::size_t t);

  // Getter for various objects and quantities
  CostStack *get_cost_stack(std::size_t t);
  CostStack *get_terminal_cost_stack();
  std::size_t get_cost_number();
  std::size_t get_size();
  std::shared_ptr<TrajOptProblem> get_problem() { return problem_; }
  std::map<std::string, std::size_t> get_cost_map() { return cost_map_; }
  RobotHandler &get_handler() { return handler_; }
  int get_nu() { return nu_; }

protected:
  // Size of the problem
  int nq_;
  int nv_;
  int ndx_;
  int nu_;

  /// Dictionnaries of cost name + cost id in the CostStack objects
  std::map<std::string, std::size_t> cost_map_;
  std::map<std::string, std::size_t> terminal_cost_map_;

  /// The robot model
  RobotHandler handler_;

  /// The reference shooting problem storing all shooting nodes
  std::shared_ptr<TrajOptProblem> problem_;

  // Vector reference for control cost
  Eigen::VectorXd control_ref_;
};

} // namespace simple_mpc
