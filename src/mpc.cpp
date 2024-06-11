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
constexpr std::size_t maxiters = 100;

MPC::MPC() {}

MPC::MPC(const MPCSettings &settings, const RobotHandler &handler,
         std::shared_ptr<Problem> &problem, const Eigen::VectorXd &x0) {
  initialize(settings, handler, problem, x0);
}

void MPC::initialize(const MPCSettings &settings, const RobotHandler &handler,
                     std::shared_ptr<Problem> &problem,
                     const Eigen::VectorXd &x0) {
  /** The posture required here is the full robot posture in the order of
   * pinicchio*/
  settings_ = settings;
  problem_ = problem;
  handler_ = handler;
  x0_ = x0;

  // designer settings
  x_internal_.resize(settings_.nq + settings_.nv);
  handler_.updateInternalData(x0);

  ref_frame_poses_.reserve(handler_.get_ee_ids().size());
  for (unsigned long i = 0; i < handler_.get_ee_ids().size(); i++) {
    ref_frame_poses_[i].assign(problem_->problem_->numSteps() + 1,
                               handler_.get_ee_pose(i));
  }

  horizon_iteration_ = 0;

  // horizon settings
  Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(settings_.nu);

  for (std::size_t i = 0; i < problem_->problem_->numSteps(); i++) {
    xs_.push_back(x0_);
    us_.push_back(zero_u);
  }
  xs_.push_back(x0_);

  solver_ = std::make_shared<SolverProxDDP>(settings_.TOL, settings_.mu_init,
                                            0., maxiters, aligator::QUIET);

  solver_->rollout_type_ = aligator::RolloutType::LINEAR;
  solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver_->force_initial_condition_ = true;
  solver_->reg_min = 1e-6;
  solver_->setNumThreads(settings_.num_threads);
  solver_->setup(*problem_->problem_);

  solver_->run(*problem_->problem_, xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];
}

void MPC::generateFullHorizon(
    const std::vector<ContactMap> &contact_phases,
    const std::vector<std::vector<Eigen::VectorXd>> &contact_forces) {
  for (std::size_t i = 0; i < contact_phases.size(); i++) {
    StageModel sm =
        problem_->create_stage(contact_phases[i], contact_forces[i]);
    full_horizon_.push_back(sm);
    full_horizon_data_.push_back(sm.createData());
  }
}

bool MPC::timeToSolveDDP(int iteration) {
  time_to_solve_ddp_ = !(iteration % settings_.Nc);
  return time_to_solve_ddp_;
}

void MPC::iterate(const Eigen::VectorXd &q_current,
                  const Eigen::VectorXd &v_current) {
  x0_ = handler_.shapeState(q_current, v_current);

  // ~~TIMING~~ //
  recedeWithCycle();
  updateSupportTiming();

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

void MPC::updateSupportTiming() {
  for (unsigned long i = 0; i < land_LF_.size(); i++)
    land_LF_[i] -= 1;
  for (unsigned long i = 0; i < land_RF_.size(); i++)
    land_RF_[i] -= 1;
  for (unsigned long i = 0; i < takeoff_LF_.size(); i++)
    takeoff_LF_[i] -= 1;
  for (unsigned long i = 0; i < takeoff_RF_.size(); i++)
    takeoff_RF_[i] -= 1;

  if (land_LF_.size() > 0 && land_LF_[0] < 0)
    land_LF_.erase(land_LF_.begin());

  if (land_RF_.size() > 0 && land_RF_[0] < 0)
    land_RF_.erase(land_RF_.begin());

  if (takeoff_LF_.size() > 0 && takeoff_LF_[0] < 0)
    takeoff_LF_.erase(takeoff_LF_.begin());

  if (takeoff_RF_.size() > 0 && takeoff_RF_[0] < 0)
    takeoff_RF_.erase(takeoff_RF_.begin());
}

void MPC::updateStepTrackerReferences() {
  for (unsigned long time = 0; time < problem_->problem_->stages_.size();
       time++) {
    problem_->set_reference_poses(time, ref_frame_poses_[time]);
  }
}

} // namespace simple_mpc
