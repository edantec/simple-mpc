///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_FWD_HPP_
#define SIMPLE_MPC_FWD_HPP_

#include <Eigen/Core>
#include <aligator/fwd.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include <aligator/modelling/multibody/frame-placement.hpp>
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/contact-map.hpp"
#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc {

using std::shared_ptr;

typedef Eigen::Matrix<double, 6, 1> eVector6;
typedef Eigen::Matrix<double, 4, 1> eVector4;
typedef Eigen::Vector3d eVector3;
typedef Eigen::Vector2d eVector2;
typedef std::shared_ptr<aligator::FramePlacementResidualTpl<double>>
    framePlacement;

// MPC
class fullDynamicsMPC;
struct Settings;
struct FullDynamicsSettings;
class RobotHandler;

} // namespace simple_mpc

#endif // SOBEC_FWD_HPP_
