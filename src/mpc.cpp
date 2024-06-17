///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/mpc.hpp"

namespace simple_mpc {
using namespace aligator;
constexpr std::size_t maxiters = 10;

MPC::MPC() {}

MPC::MPC(const MPCSettings &settings, const RobotHandler &handler,
         std::shared_ptr<Problem> &problem, const Eigen::VectorXd &x0,
         const Eigen::VectorXd &u0) {
  initialize(settings, handler, problem, x0, u0);
}

void MPC::initialize(const MPCSettings &settings, const RobotHandler &handler,
                     std::shared_ptr<Problem> &problem,
                     const Eigen::VectorXd &x0, const Eigen::VectorXd &u0) {
  /** The posture required here is the full robot posture in the order of
   * pinicchio*/
  settings_ = settings;
  problem_ = problem;
  handler_ = handler;
  x0_ = x0;
  u0_ = u0;

  // designer settings
  x_internal_.resize(settings_.nq + settings_.nv);
  // handler_.updateInternalData(x0);

  for (std::size_t i = 0; i < problem->get_size(); i++) {
    std::map<std::string, pinocchio::SE3> map_se3;
    for (std::size_t j = 0; j < handler_.get_ee_names().size(); j++) {
      map_se3.insert({handler_.get_ee_name(j), handler_.get_ee_pose(j)});
    }
    ref_frame_poses_.push_back(map_se3);
  }

  horizon_iteration_ = 0;

  for (std::size_t i = 0; i < problem_->problem_->numSteps(); i++) {
    xs_.push_back(x0_);
    us_.push_back(u0_);
  }
  xs_.push_back(x0_);

  solver_ = std::make_shared<SolverProxDDP>(settings_.TOL, settings_.mu_init,
                                            0., maxiters, aligator::VERBOSE);

  solver_->rollout_type_ = aligator::RolloutType::LINEAR;
  if (settings_.num_threads > 1) {
    solver_->linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver_->setNumThreads(settings_.num_threads);
  } else
    solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver_->force_initial_condition_ = true;
  // solver_->reg_min = 1e-6;
  solver_->setup(*problem_->problem_);

  solver_->run(*problem_->problem_, xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];
}

void MPC::generateFullHorizon(
    const std::vector<ContactMap> &contact_phases,
    const std::vector<std::map<std::string, Eigen::VectorXd>> &contact_forces) {
  for (std::size_t i = 0; i < contact_phases.size(); i++) {
    StageModel sm =
        problem_->create_stage(contact_phases[i], contact_forces[i]);
    full_horizon_.push_back(sm);
    full_horizon_data_.push_back(sm.createData());
  }
}

void MPC::iterate(const Eigen::VectorXd &q_current,
                  const Eigen::VectorXd &v_current) {
  x0_ = handler_.shapeState(q_current, v_current);

  // ~~TIMING~~ //
  recedeWithCycle();
  // updateSupportTiming();

  // ~~REFERENCES~~ //
  handler_.updateInternalData(x0_);
  // updateStepTrackerLastReference();
  updateStepTrackerReferences();

  xs_.erase(xs_.begin());
  xs_[0] = x0_;
  xs_.push_back(xs_.back());

  us_.erase(us_.begin());
  us_.push_back(us_.back());

  // ~~SOLVER~~ //
  solver_->run(*problem_->problem_, xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];
}

void MPC::recedeWithCycle() {
  if (horizon_iteration_ < full_horizon_.size()) {
    problem_->problem_->replaceStageCircular(full_horizon_[horizon_iteration_]);
    solver_->workspace_.cycleAppend(full_horizon_data_[horizon_iteration_]);
    horizon_iteration_++;
  } else {
    problem_->problem_->replaceStageCircular(problem_->problem_->stages_[0]);
    solver_->workspace_.cycleLeft();
  }
}

/* void MPC::updateSupportTiming() {
  for (auto name : handler_.get_ee_names()) {
    for (std::size_t i = 0; i < foot_land_times_.at(name).size(); i++) {
      foot_land_times_.at(name)[i] -= 1;
    }
    if (foot_land_times_.at(name).size() > 0 && foot_land_times_.at(name)[0] <
0) foot_land_times_.at(name).erase(foot_land_times_.at(name).begin()); for
(std::size_t i = 0; i < foot_takeoff_times_.at(name).size(); i++) {
      foot_takeoff_times_.at(name)[i] -= 1;
    }
    if (foot_takeoff_times_.at(name).size() > 0 &&
foot_takeoff_times_.at(name)[0] < 0)
      foot_takeoff_times_.at(name).erase(foot_takeoff_times_.at(name).begin());
    }
} */

void MPC::updateStepTrackerReferences() {
  for (unsigned long time = 0; time < problem_->problem_->stages_.size();
       time++) {
    problem_->set_reference_poses(time, ref_frame_poses_[time]);
  }
}

} // namespace simple_mpc
