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
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#ifndef ALIGATOR_PINOCCHIO_V3
#error "aligator no compile with pin v3"
#endif
#include <aligator/modelling/multibody/contact-force.hpp>

namespace simple_mpc {
using namespace aligator;
using StageModel = StageModelTpl<double>;
using CostStack = CostStackTpl<double>;
using TrajOptProblem = TrajOptProblemTpl<double>;
using CostAbstract = CostAbstractTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = dynamics::IntegratorEulerTpl<double>;
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
  createStage(const std::map<std::string, bool> &contact_phase,
              const std::map<std::string, pinocchio::SE3> &contact_pose,
              const std::map<std::string, Eigen::VectorXd> &force_refs) = 0;

  // Create the complete vector of stages from contact_sequence
  virtual std::vector<xyz::polymorphic<StageModel>> createStages(
      const std::vector<std::map<std::string, bool>> &contact_phases,
      const std::vector<std::map<std::string, pinocchio::SE3>> &contact_poses,
      const std::vector<std::map<std::string, Eigen::VectorXd>>
          &contact_forces);

  // Manage terminal cost and constraint
  virtual CostStack createTerminalCost() = 0;

  virtual void updateTerminalConstraint() = 0;

  virtual void createTerminalConstraint() = 0;

  // Setter and getter for poses reference
  virtual void setReferencePose(const std::size_t t, const std::string &ee_name,
                                const pinocchio::SE3 &pose_ref) = 0;
  virtual void
  setReferencePoses(const std::size_t t,
                    const std::map<std::string, pinocchio::SE3> &pose_refs) = 0;
  virtual void setTerminalReferencePose(const std::string &ee_name,
                                        const pinocchio::SE3 &pose_ref) = 0;
  virtual const pinocchio::SE3 getReferencePose(const std::size_t t,
                                                const std::string &ee_name) = 0;

  // Setter and getter for forces reference
  virtual void setReferenceForces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) = 0;
  virtual void setReferenceForce(const std::size_t t,
                                 const std::string &ee_name,
                                 const Eigen::VectorXd &force_ref) = 0;
  virtual const Eigen::VectorXd
  getReferenceForce(const std::size_t t, const std::string &ee_name) = 0;
  virtual const Eigen::VectorXd getProblemState() = 0;

  /// Common functions for all problems

  // Create one TrajOptProblem from contact sequence
  void createProblem(const Eigen::VectorXd &x0, const size_t horizon,
                     const int force_size, const double gravity);

  // Setter and getter for control reference
  void setReferenceControl(const std::size_t t, const Eigen::VectorXd &u_ref);

  const Eigen::VectorXd getReferenceControl(const std::size_t t);

  // Getter for various objects and quantities
  CostStack *getCostStack(std::size_t t);
  CostStack *getTerminalCostStack();
  std::size_t getCostNumber();
  std::size_t getSize();
  std::shared_ptr<TrajOptProblem> getProblem() { return problem_; }
  RobotHandler &getHandler() { return handler_; }
  int getNu() { return nu_; }

protected:
  // Size of the problem
  int nq_;
  int nv_;
  int ndx_;
  int nu_;
  bool problem_initialized_ = false;
  bool terminal_constraint_ = false;

  /// The robot model
  RobotHandler handler_;

  /// The reference shooting problem storing all shooting nodes
  std::shared_ptr<TrajOptProblem> problem_;

  // Vector reference for control cost
  Eigen::VectorXd control_ref_;
};

} // namespace simple_mpc
