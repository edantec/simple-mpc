///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_MPC_HPP_
#define SIMPLE_MPC_MPC_HPP_

#include "aligator/modelling/contact-map.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"
#include "aligator/modelling/multibody/centroidal-momentum-derivative.hpp"
#include "aligator/modelling/multibody/centroidal-momentum.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include <pinocchio/algorithm/proximal.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/foot-trajectory.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using StageModel = StageModelTpl<double>;
using StageData = StageDataTpl<double>;
using CostStack = CostStackTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using KinodynamicsFwdDynamics = dynamics::KinodynamicsFwdDynamicsTpl<double>;
using ODEAbstract = dynamics::ODEAbstractTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using ContactMap = ContactMapTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using TrajOptProblem = TrajOptProblemTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using CentroidalMomentumDerivativeResidual =
    CentroidalMomentumDerivativeResidualTpl<double>;
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
  double dt = 0.01;
};
class MPC {

protected:
  enum LocomotionType { WALKING, STANDING, MOTION };

  MPCSettings settings_;
  std::shared_ptr<Problem> problem_;
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
  Eigen::VectorXd velocity_base_;
  Eigen::VectorXd pose_base_;
  Eigen::Vector3d next_pose_;
  Eigen::Vector3d twist_vect_;

public:
  MPC();
  MPC(const MPCSettings &settings, std::shared_ptr<Problem> problem);
  void initialize(const MPCSettings &settings,
                  std::shared_ptr<Problem> problem);

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
                                        const std::string &ee_name);

  void setVelocityBase(const Eigen::VectorXd &velocity_base) {
    velocity_base_ = velocity_base;
  };
  void setPoseBase(const Eigen::VectorXd pose_ref);
  const Eigen::VectorXd getPoseBase(const std::size_t t) {
    return problem_->getPoseBase(t);
  }

  // getters and setters
  MPCSettings &getSettings() { return settings_; }

  std::shared_ptr<Problem> getProblem() { return problem_; }
  TrajOptProblem &getTrajOptProblem() { return *problem_->getProblem(); }
  SolverProxDDP &getSolver() { return *solver_; }
  RobotHandler &getHandler() { return problem_->getHandler(); }
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

  void switchToWalk(const Eigen::VectorXd &velocity_base);

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

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
