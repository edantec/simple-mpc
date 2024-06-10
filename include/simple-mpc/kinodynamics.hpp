///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef SIMPLE_MPC_KINODYNAMICS_HPP_
#define SIMPLE_MPC_KINODYNAMICS_HPP_

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/contact-map.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace aligator;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using IntegratorSemiImplEuler =
    aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using KinodynamicsFwdDynamics =
    aligator::dynamics::KinodynamicsFwdDynamicsTpl<double>;
using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using ContactMap = aligator::ContactMapTpl<double>;
using FramePlacementResidual = aligator::FramePlacementResidualTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;
using CentroidalMomentumResidual =
    aligator::CentroidalMomentumResidualTpl<double>;
using CentroidalMomentumDerivativeResidual =
    aligator::CentroidalMomentumDerivativeResidualTpl<double>;

/**
 * @brief Build a full dynamics problem
 */

struct KinodynamicsSettings : public Settings {
  Eigen::VectorXd w_cent;
  Eigen::Vector3d w_centder;

  KinodynamicsSettings();
  virtual ~KinodynamicsSettings() {}
};

class KinodynamicsProblem : public Problem {
  using Base = Problem;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KinodynamicsProblem();
  KinodynamicsProblem(const KinodynamicsSettings &settings,
                      const RobotHandler &handler);

  virtual ~KinodynamicsProblem(){};

  StageModel create_stage(const ContactMap &contact_map,
                          const std::vector<Eigen::VectorXd> &force_refs);
  virtual void set_reference_control(const std::size_t i,
                                     const Eigen::VectorXd &u_ref);
  CostStack create_terminal_cost();

  Eigen::VectorXd control_ref_;

  /// @brief Parameters to tune the algorithm, given at init.
  // KinodynamicsSettings settings_;

protected:
  KinodynamicsSettings settings_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
