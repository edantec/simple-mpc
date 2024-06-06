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
         std::shared_ptr<TrajOptProblem> &problem,
         const Eigen::VectorXd &x0) {
  initialize(settings, handler, problem, x0);
}

void MPC::initialize(const MPCSettings &settings, const RobotHandler &handler, 
                     std::shared_ptr<TrajOptProblem> &problem, const Eigen::VectorXd &x0) {
  /** The posture required here is the full robot posture in the order of
   * pinicchio*/
  settings_ = settings;
  problem_ = problem;
  handler_ = handler;
  x0_ = x0;

  // designer settings
  x_internal_.resize(settings_.nq + settings_.nv);
  handler_.updateInternalData(x0);
  
  ref_frame_poses_.reserve(handler_.get_frame_ids().size());
  for (unsigned long i = 0; i < handler_.get_frame_ids().size(); i++ ) {
    ref_frame_poses_[i].assign(problem_->numSteps() + 1, handler_.get_frame_pose(i));
  }

  horizon_iteration_ = 0;

  // horizon settings
  std::vector<Eigen::VectorXd> xs_init;
  std::vector<Eigen::VectorXd> us_init;
  Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(settings_.nu);

  for (std::size_t i = 0; i < problem_->numSteps(); i++) {
    xs_init.push_back(x0_);
    us_init.push_back(zero_u);
  }
  xs_init.push_back(x0_);

  SolverProxDDPTpl<double> solver(settings_.TOL, settings_.mu_init, 0., maxiters, aligator::QUIET);

  solver.rollout_type_ = aligator::RolloutType::LINEAR;
  solver.linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver.force_initial_condition_ = true;
  solver.reg_min = 1e-6;
  solver.setNumThreads(settings_.num_threads);
  solver.setup(*problem_);

}

void MPC::setForceAlongHorizon() {
  // Set force reference
  Eigen::VectorXd wrench_reference_left(6);
  Eigen::VectorXd wrench_reference_right(6);
  double ref_force;
  double mean_force = settings_.support_force / 2.;
  double min_min_force = settings_.min_force / 100.0;
  double max_min_force = settings_.support_force - settings_.min_force;
  wrench_reference_left << 0, 0, mean_force, 0, 0, 0;
  wrench_reference_right << 0, 0, mean_force, 0, 0, 0;

  // Set force reference for first cycle
  for (int i = 0; i < settings_.TdoubleSupport; i++) {
    ref_force = mean_force * (settings_.TdoubleSupport - i - 1) / static_cast<double>(settings_.TdoubleSupport) +
                min_min_force * (i + 1) / static_cast<double>(settings_.TdoubleSupport);
    wrench_reference_right[2] = ref_force;
    wrench_reference_left[2] = settings_.support_force - ref_force;
    fullHorizon_.setWrenchReference(i, "wrench_LF", wrench_reference_left);
    fullHorizon_.setWrenchReference(i, "wrench_RF", wrench_reference_right);
  }
  // Set force reference for following cycles
  for (int j = 1; j < settings_.totalSteps; j++) {
    for (int i = 0; i < settings_.TdoubleSupport; i++) {
      ref_force = max_min_force * (settings_.TdoubleSupport - i - 1) / static_cast<double>(settings_.TdoubleSupport) +
                  min_min_force * (i + 1) / static_cast<double>(settings_.TdoubleSupport);
      if (j % 2 == 0) {
        wrench_reference_right[2] = ref_force;
        wrench_reference_left[2] = settings_.support_force - ref_force;
      } else {
        wrench_reference_left[2] = ref_force;
        wrench_reference_right[2] = settings_.support_force - ref_force;
      }
      fullHorizon_.setWrenchReference(i + j * settings_.Tstep, "wrench_LF", wrench_reference_left);
      fullHorizon_.setWrenchReference(i + j * settings_.Tstep, "wrench_RF", wrench_reference_right);
    }
  }
  // Set force reference for last cycle
  for (int i = 0; i < settings_.TdoubleSupport; i++) {
    ref_force = max_min_force * (settings_.TdoubleSupport - i - 1) / static_cast<double>(settings_.TdoubleSupport) +
                mean_force * (i + 1) / static_cast<double>(settings_.TdoubleSupport);
    if (settings_.totalSteps % 2 == 0) {
      wrench_reference_right[2] = ref_force;
      wrench_reference_left[2] = settings_.support_force - ref_force;
    } else {
      wrench_reference_right[2] = ref_force;
      wrench_reference_left[2] = settings_.support_force - ref_force;
    }
    fullHorizon_.setWrenchReference(i + settings_.totalSteps * settings_.Tstep, "wrench_LF", wrench_reference_left);
    fullHorizon_.setWrenchReference(i + settings_.totalSteps * settings_.Tstep, "wrench_RF", wrench_reference_right);
  }
}

std::vector<Support> WBCHorizon::generateSupportCycle() {
  std::vector<Support> cycle;

  for (int j = 0; j < settings_.totalSteps; j++) {
    if (j % 2 == 0) {
      takeoff_RF_.push_back(j * settings_.Tstep + settings_.TdoubleSupport + settings_.T);
      land_RF_.push_back((j + 1) * settings_.Tstep + settings_.T);
    } else {
      takeoff_LF_.push_back(j * settings_.Tstep + settings_.TdoubleSupport + settings_.T);
      land_LF_.push_back((j + 1) * settings_.Tstep + settings_.T);
    }
    for (int i = 0; i < settings_.Tstep; i++) {
      if (i < settings_.TdoubleSupport)
        cycle.push_back(DOUBLE);
      else {
        if (j % 2 == 0)
          cycle.push_back(LEFT);
        else
          cycle.push_back(RIGHT);
      }
    }
  }
  for (int j = 0; j < settings_.T + settings_.TdoubleSupport; j++) {
    cycle.push_back(DOUBLE);
  }
  return cycle;
}

void MPC::generateFullHorizon(ModelMaker &mm, const Experiment &experiment) {
  std::vector<Support> cycle = generateSupportCycle();
  std::vector<AMA> cyclicModels;
  cyclicModels = mm.formulateHorizon(cycle, experiment);
  HorizonManagerSettings names = {designer_.get_LF_name(), designer_.get_RF_name()};
  fullHorizon_ = HorizonManager(names, x0_, cyclicModels, cyclicModels.back());
  setForceAlongHorizon();
}

bool MPC::timeToSolveDDP(int iteration) {
  time_to_solve_ddp_ = !(iteration % settings_.Nc);
  return time_to_solve_ddp_;
}

void MPC::iterate(const Eigen::VectorXd &q_current, const Eigen::VectorXd &v_current, bool is_feasible) {
  x0_ = designer_.shapeState(q_current, v_current);

  // ~~TIMING~~ //
  recedeWithCycle();
  updateSupportTiming();

  // ~~REFERENCES~~ //
  designer_.updateReducedModel(x0_);
  // updateStepTrackerLastReference();
  updateStepTrackerReferences();

  // ~~SOLVER~~ //
  horizon_.solve(x0_, settings_.ddpIteration, is_feasible);
}

void MPC::iterate(int iteration, const Eigen::VectorXd &q_current, const Eigen::VectorXd &v_current,
                         bool is_feasible) {
  if (timeToSolveDDP(iteration)) {
    iterate(q_current, v_current, is_feasible);
  } else
    x0_ = designer_.shapeState(q_current, v_current);
}

void MPC::updateStepTrackerReferences() {
  for (unsigned long time = 0; time < horizon_.size(); time++) {
    horizon_.setPoseReference(time, "placement_LF", getPoseRef_LF(time));
    horizon_.setPoseReference(time, "placement_RF", getPoseRef_RF(time));
    horizon_.setActuationReference(time, "actuationDrift", getTorqueRef(time));
    ///@todo: the names must be provided by the user
  }
  horizon_.setTerminalPoseReference("placement_LF", getPoseRef_LF(horizon_.size()));
  horizon_.setTerminalPoseReference("placement_RF", getPoseRef_RF(horizon_.size()));

  if (horizon_.contacts(horizon_.size() - 1)->getContactStatus(designer_.get_LF_name()) and
      horizon_.contacts(horizon_.size() - 1)->getContactStatus(designer_.get_RF_name())) {
    ref_dcm_ = (getPoseRef_LF(horizon_.size()).translation() + getPoseRef_RF(horizon_.size()).translation()) / 2;
  } else if (horizon_.contacts(horizon_.size() - 1)->getContactStatus(designer_.get_LF_name())) {
    ref_dcm_ = getPoseRef_LF(horizon_.size()).translation();
  } else if (horizon_.contacts(horizon_.size() - 1)->getContactStatus(designer_.get_RF_name())) {
    ref_dcm_ = getPoseRef_RF(horizon_.size()).translation();
  }
  ref_dcm_[2] = 0.87;
  horizon_.setTerminalDCMReference("DCM", ref_dcm_);
}

void MPC::updateStepTrackerLastReference() {
  horizon_.setPoseReference(horizon_.size() - 1, "placement_LF", getPoseRef_LF(horizon_.size() - 1));
  horizon_.setPoseReference(horizon_.size() - 1, "placement_RF", getPoseRef_RF(horizon_.size() - 1));
  horizon_.setTerminalPoseReference("placement_LF", getPoseRef_LF(horizon_.size()));
  horizon_.setTerminalPoseReference("placement_RF", getPoseRef_RF(horizon_.size()));
  ref_LF_poses_.erase(ref_LF_poses_.begin());
  ref_LF_poses_.push_back(ref_LF_poses_[horizon_.size() - 1]);
  ref_RF_poses_.erase(ref_RF_poses_.begin());
  ref_RF_poses_.push_back(ref_RF_poses_[horizon_.size() - 1]);
}

void MPC::recedeWithCycle() {
  if (horizon_iteration_ < fullHorizon_.size()) {
    horizon_.recede(fullHorizon_.ama(horizon_iteration_), fullHorizon_.ada(horizon_iteration_));
    horizon_iteration_++;
  } else
    horizon_.recede();
}

void MPC::goToNextDoubleSupport() {
  while (horizon_.supportSize(0) != 2 and horizon_iteration_ < fullHorizon_.size()) {
    horizon_.recede(fullHorizon_.ama(horizon_iteration_), fullHorizon_.ada(horizon_iteration_));
    horizon_iteration_++;
    updateSupportTiming();
  }
  horizon_.recede(fullHorizon_.ama(horizon_iteration_), fullHorizon_.ada(horizon_iteration_));
  horizon_iteration_++;
  updateSupportTiming();
}

void MPC::updateSupportTiming() {
  for (unsigned long i = 0; i < land_LF_.size(); i++) land_LF_[i] -= 1;
  for (unsigned long i = 0; i < land_RF_.size(); i++) land_RF_[i] -= 1;
  for (unsigned long i = 0; i < takeoff_LF_.size(); i++) takeoff_LF_[i] -= 1;
  for (unsigned long i = 0; i < takeoff_RF_.size(); i++) takeoff_RF_[i] -= 1;

  if (land_LF_.size() > 0 && land_LF_[0] < 0) land_LF_.erase(land_LF_.begin());

  if (land_RF_.size() > 0 && land_RF_[0] < 0) land_RF_.erase(land_RF_.begin());

  if (takeoff_LF_.size() > 0 && takeoff_LF_[0] < 0) takeoff_LF_.erase(takeoff_LF_.begin());

  if (takeoff_RF_.size() > 0 && takeoff_RF_[0] < 0) takeoff_RF_.erase(takeoff_RF_.begin());
}

} // namespace simple_mpc
