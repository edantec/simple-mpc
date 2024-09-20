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
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
using namespace aligator;
constexpr std::size_t maxiters = 10;

MPC::MPC() {}

MPC::MPC(const MPCSettings &settings, std::shared_ptr<Problem> problem) {
  initialize(settings, problem);
}

void MPC::initialize(const MPCSettings &settings,
                     std::shared_ptr<Problem> problem) {
  settings_ = settings;
  problem_ = problem;
  horizon_iteration_ = 0;

  std::map<std::string, Eigen::Vector3d> initial_poses;
  Eigen::Vector3d rel_trans;
  rel_trans << settings_.x_translation, settings_.y_translation, 0;
  for (auto const &name : problem_->getHandler().getFeetNames()) {
    initial_poses.insert(
        {name, problem_->getHandler().getFootPose(name).translation()});
    relative_translations_.insert({name, rel_trans});
  }
  foot_trajectories_ =
      FootTrajectory(initial_poses, settings_.swing_apex, settings_.T_fly,
                     settings_.T_contact, settings_.T);

  foot_trajectories_.updateForward(relative_translations_, settings.swing_apex);
  x0_ = problem_->getProblemState();

  solver_ = std::make_shared<SolverProxDDP>(settings_.TOL, settings_.mu_init,
                                            0., maxiters, aligator::QUIET);
  solver_->rollout_type_ = aligator::RolloutType::LINEAR;
  if (settings_.num_threads > 1) {
    solver_->linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver_->setNumThreads(settings_.num_threads);
  } else
    solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver_->force_initial_condition_ = true;
  // solver_->reg_min = 1e-6;

  ee_names_ = problem_->getHandler().getFeetNames();

  for (std::size_t i = 0; i < problem_->getProblem()->numSteps(); i++) {
    xs_.push_back(x0_);
    us_.push_back(problem_->getReferenceControl(0));
  }
  xs_.push_back(x0_);

  solver_->setup(*problem_->getProblem());
  solver_->run(*problem_->getProblem(), xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];

  solver_->max_iters = settings_.max_iters;
}

void MPC::generateFullHorizon(
    const std::vector<std::map<std::string, bool>> &contact_states) {
  for (auto const &name : ee_names_) {
    foot_takeoff_times_.insert({name, std::vector<int>()});
    foot_land_times_.insert({name, std::vector<int>()});
  }
  for (size_t i = 1; i < contact_states.size(); i++) {
    for (auto const &name : ee_names_) {
      if (!contact_states[i].at(name) && contact_states[i - 1].at(name)) {
        foot_takeoff_times_.at(name).push_back((int)(i + problem_->getSize()));
      }
      if (contact_states[i].at(name) && !contact_states[i - 1].at(name)) {
        foot_land_times_.at(name).push_back((int)(i + problem_->getSize()));
      }
    }
  }
  for (auto const &state : contact_states) {
    int active_contacts = 0;
    for (auto const &contact : state) {
      if (contact.second)
        active_contacts += 1;
    }

    Eigen::VectorXd force_ref(
        problem_->getReferenceForce(0, problem_->getHandler().getFootName(0)));
    Eigen::VectorXd force_zero(
        problem_->getReferenceForce(0, problem_->getHandler().getFootName(0)));
    force_ref.setZero();
    force_zero.setZero();
    force_ref[2] = settings_.support_force / active_contacts;

    aligator::StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
    std::map<std::string, Eigen::VectorXd> force_map;
    std::vector<bool> contact_bools;

    for (auto const &name : ee_names_) {
      contact_poses.push_back(
          problem_->getHandler().getFootPose(name).translation());
      contact_bools.push_back(state.at(name));
      if (state.at(name))
        force_map.insert({name, force_ref});
      else
        force_map.insert({name, force_zero});
    }

    ContactMap contact_map(ee_names_, contact_bools, contact_poses);

    StageModel sm = problem_->createStage(contact_map, force_map);
    full_horizon_.push_back(sm);
    full_horizon_data_.push_back(sm.createData());
  }
}

void MPC::iterate(const Eigen::VectorXd &q_current,
                  const Eigen::VectorXd &v_current) {

  problem_->getHandler().updateState(q_current, v_current, false);

  // ~~TIMING~~ //
  recedeWithCycle();
  updateSupportTiming();

  // ~~REFERENCES~~ //
  x0_ << q_current, v_current;
  updateStepTrackerReferences();

  xs_.erase(xs_.begin());
  xs_[0] = x0_;
  xs_.push_back(xs_.back());

  us_.erase(us_.begin());
  us_.push_back(us_.back());
  problem_->getProblem()->setInitState(x0_);

  // ~~SOLVER~~ //
  solver_->run(*problem_->getProblem(), xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];
}

void MPC::recedeWithCycle() {
  if (horizon_iteration_ < full_horizon_.size()) {
    problem_->getProblem()->replaceStageCircular(
        full_horizon_[horizon_iteration_]);
    solver_->workspace_.cycleAppend(full_horizon_data_[horizon_iteration_]);
    horizon_iteration_++;
  } else {
    problem_->getProblem()->replaceStageCircular(
        problem_->getProblem()->stages_[0]);
    solver_->workspace_.cycleLeft();
  }
}

void MPC::updateSupportTiming() {
  RobotHandler handler = problem_->getHandler();
  for (auto const &name : ee_names_) {
    for (size_t i = 0; i < foot_land_times_.at(name).size(); i++) {
      foot_land_times_.at(name)[i] -= 1;
      if (foot_land_times_.at(name)[0] < 0)
        foot_land_times_.at(name).erase(foot_land_times_.at(name).begin());
    }
    for (size_t i = 0; i < foot_takeoff_times_.at(name).size(); i++) {
      foot_takeoff_times_.at(name)[i] -= 1;
      if (foot_takeoff_times_.at(name)[0] < 0)
        foot_takeoff_times_.at(name).erase(
            foot_takeoff_times_.at(name).begin());
    }
  }
}

void MPC::updateStepTrackerReferences() {
  for (auto const &name : ee_names_) {
    int foot_land_time = -1;
    int foot_takeoff_time = -1;
    if (!foot_land_times_.at(name).empty())
      foot_land_time = foot_land_times_.at(name)[0];
    if (!foot_takeoff_times_.at(name).empty())
      foot_takeoff_time = foot_takeoff_times_.at(name)[0];

    foot_trajectories_.updateTrajectory(
        foot_takeoff_time, foot_land_time,
        problem_->getHandler().getFootPose(name).translation(), name);

    for (unsigned long time = 0; time < problem_->getProblem()->stages_.size();
         time++) {
      pinocchio::SE3 pose = pinocchio::SE3::Identity();
      pose.translation() = foot_trajectories_.getReference(name)[time];
      setReferencePose(time, name, pose);
    }
    pinocchio::SE3 pose = pinocchio::SE3::Identity();
    pose.translation() = foot_trajectories_.getReference(
        name)[problem_->getProblem()->stages_.size() - 1];
    setTerminalReferencePose(name, pose);
  }
}

void MPC::setReferencePose(const std::size_t t, const std::string &ee_name,
                           const pinocchio::SE3 &pose_ref) {
  problem_->setReferencePose(t, ee_name, pose_ref);
}

void MPC::setTerminalReferencePose(const std::string &ee_name,
                                   const pinocchio::SE3 &pose_ref) {
  problem_->setTerminalReferencePose(ee_name, pose_ref);
}

const pinocchio::SE3 MPC::getReferencePose(const std::size_t t,
                                           const std::string &ee_name) {
  return problem_->getReferencePose(t, ee_name);
}

void MPC::setRelativeTranslation(
    const std::map<std::string, Eigen::Vector3d> &relative_translations,
    const double swing_apex) {
  relative_translations_ = relative_translations;
  foot_trajectories_.updateForward(relative_translations_, swing_apex);
}

} // namespace simple_mpc
