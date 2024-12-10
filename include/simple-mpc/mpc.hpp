///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "simple-mpc/deprecated.hpp"
#include "simple-mpc/foot-trajectory.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
using namespace aligator;
using StageData = StageDataTpl<double>;
using SolverProxDDP = SolverProxDDPTpl<double>;
/**
 * @brief Build a MPC object holding an instance
 * of a trajectory optimization problem
 */

struct MPCSettings {
public:
  // Step-related quantities
  double swing_apex = 0.15;

  // Force parameters
  double support_force = 1000;

  // Solver-related quantities
  double TOL = 1e-4;
  double mu_init = 1e-8;
  std::size_t max_iters = 1;
  std::size_t num_threads = 2;
  int ddpIteration = 1;

  // Timings
  int T_fly = 80;
  int T_contact = 20;
  size_t T = 100;
  double timestep = 0.01;
};
class MPC {

protected:
  enum LocomotionType { WALKING, STANDING, MOTION };

  std::vector<std::map<std::string, bool>> contact_states_;
  std::vector<std::shared_ptr<StageModel>> cycle_horizon_;
  std::vector<std::shared_ptr<StageData>> cycle_horizon_data_;
  std::vector<std::shared_ptr<StageModel>> one_horizon_;
  std::vector<std::shared_ptr<StageData>> one_horizon_data_;
  std::vector<std::shared_ptr<StageModel>> standing_horizon_;
  std::vector<std::shared_ptr<StageData>> standing_horizon_data_;
  std::shared_ptr<SolverProxDDP> solver_;
  FootTrajectory foot_trajectories_;
  std::map<std::string, pinocchio::SE3> relative_feet_poses_;
  // INTERNAL UPDATING function
  void updateStepTrackerReferences();

  // Memory preallocations:
  std::vector<unsigned long> controlled_joints_id_;
  std::vector<std::string> ee_names_;
  Eigen::VectorXd x_internal_;
  bool time_to_solve_ddp_ = false;
  Eigen::Vector3d com0_;
  LocomotionType now_;

public:
  Vector6d velocity_base_;
  Vector7d pose_base_;
  Eigen::Vector3d next_pose_;
  Eigen::Vector3d twist_vect_;
  MPCSettings settings_;
  std::shared_ptr<OCPHandler> ocp_handler_;

  MPC();
  MPC(const MPCSettings &settings, std::shared_ptr<OCPHandler> problem);
  void initialize(const MPCSettings &settings,
                  std::shared_ptr<OCPHandler> problem);

  // Generate the cycle walking problem along which we will iterate
  // the receding horizon
  void generateCycleHorizon(
      const std::vector<std::map<std::string, bool>> &contact_states);

  // Perform one iteration of MPC
  void iterate(const Eigen::VectorXd &q_current,
               const Eigen::VectorXd &v_current);

  void updateCycleTiming(const bool updateOnlyHorizon);

  // Recede the horizon
  void recedeWithCycle();

  // Getters and setters
  void setReferencePose(const std::size_t t, const std::string &ee_name,
                        const pinocchio::SE3 &pose_ref);

  void setTerminalReferencePose(const std::string &ee_name,
                                const pinocchio::SE3 &pose_ref);

  const pinocchio::SE3 getReferencePose(const std::size_t t,
                                        const std::string &ee_name) const;

  SIMPLE_MPC_DEPRECATED void setVelocityBase(const Vector6d &v) {
    velocity_base_ = v;
  }

  void setPoseBaseFromSE3(const pin::SE3 &pose_ref) {
    Eigen::Map<pin::SE3::Quaternion> q{pose_base_.tail<4>().data()};
    pose_base_.head<3>() = pose_ref.translation();
    q = pose_ref.rotation();
  }
  SIMPLE_MPC_DEPRECATED void setPoseBase(const Vector7d &pose_ref) {
    pose_base_ = pose_ref;
  }

  ConstVectorRef getPoseBase(const std::size_t t) const {
    return ocp_handler_->getPoseBase(t);
  }

  // getters and setters
  TrajOptProblem &getTrajOptProblem() { return ocp_handler_->getProblem(); }

  SolverProxDDP &getSolver() { return *solver_; }

  RobotHandler &getHandler() { return ocp_handler_->getHandler(); }

  std::vector<std::shared_ptr<StageModel>> &getCycleHorizon() {
    return cycle_horizon_;
  }
  bool getCyclingContactState(const std::size_t t, const std::string &ee_name);
  int getFootTakeoffCycle(const std::string &ee_name) {
    if (foot_takeoff_times_.at(ee_name).empty()) {
      return -1;
    } else {
      return foot_takeoff_times_.at(ee_name)[0];
    }
  }
  int getFootLandCycle(const std::string &ee_name) {
    if (foot_land_times_.at(ee_name).empty()) {
      return -1;
    } else {
      return foot_land_times_.at(ee_name)[0];
    }
  }

  void switchToWalk(const Vector6d &velocity_base);

  void switchToStand();

  // Footstep timings for each end effector
  std::map<std::string, std::vector<int>> foot_takeoff_times_, foot_land_times_;

  // Solution vectors for state and control
  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> us_;

  // Riccati gains
  std::vector<Eigen::MatrixXd> Ks_;

  // Initial quantities
  Eigen::VectorXd x0_;
  Eigen::VectorXd u0_;
};

} // namespace simple_mpc
