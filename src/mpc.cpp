///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-data.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/mpc.hpp"
#include "simple-mpc/robot-handler.hpp"
#include <chrono>

namespace simple_mpc {
using namespace aligator;
constexpr std::size_t maxiters = 100;

MPC::MPC() {}

MPC::MPC(const MPCSettings &settings, std::shared_ptr<Problem> problem) {
  initialize(settings, problem);
}

void MPC::initialize(const MPCSettings &settings,
                     std::shared_ptr<Problem> problem) {
  settings_ = settings;
  problem_ = problem;
  std::map<std::string, Eigen::Vector3d> starting_poses;
  for (auto const &name : problem_->getHandler().getFeetNames()) {
    starting_poses.insert(
        {name, problem_->getHandler().getFootPose(name).translation()});

    relative_feet_poses_.insert(
        {name, problem_->getHandler().getRootFrame().inverse() *
                   problem_->getHandler().getFootPose(name)});
  }
  foot_trajectories_ =
      FootTrajectory(starting_poses, settings_.swing_apex, settings_.T_fly,
                     settings_.T_contact, settings_.T);

  foot_trajectories_.updateForward(settings.swing_apex);
  x0_ = problem_->getProblemState();

  solver_ = std::make_shared<SolverProxDDP>(settings_.TOL, settings_.mu_init,
                                            maxiters, aligator::QUIET);
  solver_->rollout_type_ = aligator::RolloutType::LINEAR;

  if (settings_.num_threads > 1) {
    solver_->linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver_->setNumThreads(settings_.num_threads);
  } else
    solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver_->force_initial_condition_ = true;
  // solver_->reg_min = 1e-6;

  ee_names_ = problem_->getHandler().getFeetNames();
  Eigen::VectorXd force_ref(
      problem_->getReferenceForce(0, problem_->getHandler().getFootName(0)));

  std::map<std::string, bool> contact_states;
  std::map<std::string, bool> land_constraint;
  std::map<std::string, pinocchio::SE3> contact_poses;
  std::map<std::string, Eigen::VectorXd> force_map;

  for (auto const &name : ee_names_) {
    contact_states.insert({name, true});
    land_constraint.insert({name, false});
    contact_poses.insert({name, problem_->getHandler().getFootPose(name)});
    force_map.insert({name, force_ref});
  }

  for (std::size_t i = 0; i < problem_->getProblem()->numSteps(); i++) {
    xs_.push_back(x0_);
    us_.push_back(problem_->getReferenceControl(0));

    std::shared_ptr<StageModel> sm =
        std::make_shared<StageModel>(problem_->createStage(
            contact_states, contact_poses, force_map, land_constraint));
    standing_horizon_.push_back(sm);
    standing_horizon_data_.push_back(sm->createData());
  }
  xs_.push_back(x0_);

  solver_->setup(*problem_->getProblem());
  solver_->run(*problem_->getProblem(), xs_, us_);

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];

  solver_->max_iters = settings_.max_iters;

  com0_ = problem_->getHandler().getComPosition();
  now_ = WALKING;
  velocity_base_.resize(6);
  velocity_base_.setZero();
  next_pose_.setZero();
  twist_vect_.setZero();
}

void MPC::generateCycleHorizon(
    const std::vector<std::map<std::string, bool>> &contact_states) {
  contact_states_ = contact_states;
  for (auto const &name : ee_names_) {
    foot_takeoff_times_.insert({name, std::vector<int>()});
    foot_land_times_.insert({name, std::vector<int>()});
    for (size_t i = 1; i < contact_states.size(); i++) {
      if (!contact_states[i].at(name) and contact_states[i - 1].at(name)) {
        foot_takeoff_times_.at(name).push_back((int)(i + problem_->getSize()));
      }
      if (contact_states[i].at(name) and !contact_states[i - 1].at(name)) {
        foot_land_times_.at(name).push_back((int)(i + problem_->getSize()));
      }
    }
    if (contact_states.back().at(name) and !contact_states[0].at(name))
      foot_takeoff_times_.at(name).push_back(
          (int)(contact_states.size() - 1 + problem_->getSize()));
    if (!contact_states.back().at(name) and contact_states[0].at(name))
      foot_land_times_.at(name).push_back(
          (int)(contact_states.size() - 1 + problem_->getSize()));
  }
  std::map<std::string, bool> previous_contacts;
  for (auto const &name : ee_names_) {
    previous_contacts.insert({name, true});
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

    std::map<std::string, pinocchio::SE3> contact_poses;
    std::map<std::string, Eigen::VectorXd> force_map;

    for (auto const &name : ee_names_) {
      contact_poses.insert({name, problem_->getHandler().getFootPose(name)});
      if (state.at(name))
        force_map.insert({name, force_ref});
      else
        force_map.insert({name, force_zero});
    }
    std::map<std::string, bool> land_contacts;
    for (auto const &name : ee_names_) {
      if (!previous_contacts.at(name) and state.at(name)) {
        land_contacts.insert({name, true});
      } else {
        land_contacts.insert({name, false});
      }
    }

    std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(
        problem_->createStage(state, contact_poses, force_map, land_contacts));
    cycle_horizon_.push_back(sm);
    cycle_horizon_data_.push_back(sm->createData());
    previous_contacts = state;
  }
}

void MPC::iterate(const Eigen::VectorXd &q_current,
                  const Eigen::VectorXd &v_current) {

  problem_->getHandler().updateState(q_current, v_current, false);

  // ~~TIMING~~ //
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  recedeWithCycle();
  /* std::chrono::steady_clock::time_point end =
  std::chrono::steady_clock::now(); std::cout << "recedeCycle = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl; */

  // ~~REFERENCES~~ //
  updateStepTrackerReferences();

  x0_ << problem_->getProblemState();
  xs_.erase(xs_.begin());
  xs_[0] = x0_;
  xs_.push_back(xs_.back());

  us_.erase(us_.begin());
  us_.push_back(us_.back());

  problem_->getProblem()->setInitState(x0_);

  // ~~SOLVER~~ //
  std::chrono::steady_clock::time_point begin5 =
      std::chrono::steady_clock::now();
  solver_->run(*problem_->getProblem(), xs_, us_);
  /* std::chrono::steady_clock::time_point end5 =
  std::chrono::steady_clock::now(); std::cout << "solve = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end5 -
                                                                     begin5)
                   .count()
            << "[ms]" << std::endl; */

  xs_ = solver_->results_.xs;
  us_ = solver_->results_.us;
  K0_ = solver_->results_.getCtrlFeedbacks()[0];
}

void MPC::recedeWithCycle() {
  if (now_ == WALKING or
      problem_->getContactSupport(settings_.T - 1) < ee_names_.size()) {

    problem_->getProblem()->replaceStageCircular(*cycle_horizon_[0]);
    solver_->cycleProblem(*problem_->getProblem(), cycle_horizon_data_[0]);

    rotate_vec_left(cycle_horizon_);
    rotate_vec_left(cycle_horizon_data_);
    rotate_vec_left(contact_states_);
    for (auto const &name : ee_names_) {
      if (!contact_states_[contact_states_.size() - 1].at(name) and
          contact_states_[contact_states_.size() - 2].at(name))
        foot_takeoff_times_.at(name).push_back(
            (int)(contact_states_.size() + problem_->getSize()));
      if (contact_states_[contact_states_.size() - 1].at(name) and
          !contact_states_[contact_states_.size() - 2].at(name))
        foot_land_times_.at(name).push_back(
            (int)(contact_states_.size() + problem_->getSize()));
    }
    updateCycleTiming(false);
  } else {
    problem_->getProblem()->replaceStageCircular(*standing_horizon_[0]);
    solver_->cycleProblem(*problem_->getProblem(), standing_horizon_data_[0]);

    rotate_vec_left(standing_horizon_);
    rotate_vec_left(standing_horizon_data_);

    updateCycleTiming(true);
  }
}

void MPC::updateCycleTiming(const bool updateOnlyHorizon) {
  for (auto const &name : ee_names_) {
    for (size_t i = 0; i < foot_land_times_.at(name).size(); i++) {
      if (!updateOnlyHorizon or
          foot_land_times_.at(name)[i] < (int)problem_->getSize())
        foot_land_times_.at(name)[i] -= 1;
    }
    if (!foot_land_times_.at(name).empty() and foot_land_times_.at(name)[0] < 0)
      foot_land_times_.at(name).erase(foot_land_times_.at(name).begin());

    for (size_t i = 0; i < foot_takeoff_times_.at(name).size(); i++)
      if (!updateOnlyHorizon or
          foot_takeoff_times_.at(name)[i] < (int)problem_->getSize()) {
        foot_takeoff_times_.at(name)[i] -= 1;
      }
    if (!foot_takeoff_times_.at(name).empty() and
        foot_takeoff_times_.at(name)[0] < 0)
      foot_takeoff_times_.at(name).erase(foot_takeoff_times_.at(name).begin());
  }
}

void MPC::updateStepTrackerReferences() {
  for (auto const &name : ee_names_) {
    int foot_land_time = -1;
    if (!foot_land_times_.at(name).empty())
      foot_land_time = foot_land_times_.at(name)[0];

    bool update = true;
    if (foot_land_time < settings_.T_fly)
      update = false;

    // Use the Raibert heuristics to compute the next foot pose
    twist_vect_[0] =
        -(problem_->getHandler().getHipPose(name).translation()[1] -
          problem_->getHandler().getRootFrame().translation()[1]);
    twist_vect_[1] = problem_->getHandler().getHipPose(name).translation()[0] -
                     problem_->getHandler().getRootFrame().translation()[0];
    next_pose_.head(2) =
        problem_->getHandler().getHipPose(name).translation().head(2);
    next_pose_.head(2) +=
        (velocity_base_.head(2) + velocity_base_[5] * twist_vect_) *
        (settings_.T_fly + settings_.T_contact) * settings_.dt;
    next_pose_[2] = problem_->getHandler().getFootPose(name).translation()[2];

    foot_trajectories_.updateTrajectory(
        update, foot_land_time,
        problem_->getHandler().getFootPose(name).translation(), next_pose_,
        name);
    pinocchio::SE3 pose = pinocchio::SE3::Identity();
    for (unsigned long time = 0; time < problem_->getSize(); time++) {
      pose.translation() = foot_trajectories_.getReference(name)[time];
      setReferencePose(time, name, pose);
    }
  }

  problem_->setVelocityBase(problem_->getSize() - 1, velocity_base_);

  Eigen::Vector3d com_ref;
  com_ref << 0, 0, 0;
  for (auto const &name : ee_names_) {
    com_ref += foot_trajectories_.getReference(name).back();
  }
  com_ref /= (double)ee_names_.size();
  com_ref[2] += com0_[2];

  // problem_->updateTerminalConstraint(com0_);
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

void MPC::switchToWalk(const Eigen::VectorXd &velocity_base) {
  now_ = WALKING;
  velocity_base_ = velocity_base;
}

void MPC::switchToStand() {
  now_ = STANDING;
  velocity_base_.setZero();
}

} // namespace simple_mpc
