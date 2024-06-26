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
#include <pinocchio/algorithm/proximal.hpp>

#include "simple-mpc/base-problem.hpp"
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
 * @brief Build a full dynamics problem
 */

struct MPCSettings {
  ///@todo: add the cost names as setting parameters.
public:
  // timing
  int totalSteps = 4;
  std::size_t T = 100;
  int ddpIteration = 1;

  double min_force = 150;
  double support_force = 1000;

  double TOL = 1e-4;
  double mu_init = 1e-8;
  std::size_t max_iters = 1;

  std::size_t num_threads = 2;
};
class MPC {
  /**
   * Form to use this class:
   * 1) The function iterate produces the torques to command.
   * 2) All cost references must be updated separtely in the control loop.
   *
   */

protected:
  MPCSettings settings_;
  std::shared_ptr<Problem> problem_;
  std::vector<StageModel> full_horizon_;
  std::vector<std::shared_ptr<StageData>> full_horizon_data_;
  std::shared_ptr<SolverProxDDP> solver_;

  // INTERNAL UPDATING functions
  void updateStepTrackerReferences();

  // References for costs:
  std::vector<std::map<std::string, pinocchio::SE3>> ref_frame_poses_;

  // Memory preallocations:
  std::vector<unsigned long> controlled_joints_id_;
  Eigen::VectorXd x_internal_;
  bool time_to_solve_ddp_ = false;

public:
  MPC();
  MPC(const MPCSettings &settings, std::shared_ptr<Problem> &problem,
      const Eigen::VectorXd &x_multibody, const Eigen::VectorXd &u0);
  MPC(const Eigen::VectorXd &x_multibody, const Eigen::VectorXd &u0);
  void initialize(const MPCSettings &settings,
                  std::shared_ptr<Problem> &problem);

  void
  generateFullHorizon(const std::vector<ContactMap> &contact_phases,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &contact_forces);

  void iterate(const Eigen::VectorXd &q_current,
               const Eigen::VectorXd &v_current);

  void recedeWithCycle();

  // void updateSupportTiming();

  // getters and setters
  MPCSettings &get_settings() { return settings_; }

  const Eigen::VectorXd &get_x0() const { return x0_; }
  void set_x0(const Eigen::VectorXd &x0) { x0_ = x0; }
  const std::vector<Eigen::VectorXd> &get_xs() const { return xs_; }
  const std::vector<Eigen::VectorXd> &get_us() const { return us_; }

  std::vector<StageModel> &get_fullHorizon() { return full_horizon_; }
  std::vector<std::shared_ptr<StageData>> &get_fullHorizonData() {
    return full_horizon_data_;
  }

  std::shared_ptr<Problem> &get_problem() { return problem_; }

  const std::map<std::string, std::vector<int>> &get_land_times() {
    return foot_land_times_;
  }
  const std::map<std::string, std::vector<int>> &get_takeoff_time() {
    return foot_takeoff_times_;
  }

  const std::size_t &get_horizon_iteration() { return horizon_iteration_; }

  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> us_;
  Eigen::MatrixXd K0_;

  Eigen::VectorXd x0_;
  Eigen::VectorXd x_multibody_;
  Eigen::VectorXd u0_;

  // timings
  std::map<std::string, std::vector<int>> foot_takeoff_times_, foot_land_times_;
  std::size_t horizon_iteration_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
