///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/fwd.hpp>

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
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
struct Settings;
struct FullDynamicsSettings;
class RobotHandler;
class FullDynamicsProblem;
class KinodynamicsProblem;
class CentroidalProblem;
class Problem;
class IDSolver;
class IKIDSolver;

} // namespace simple_mpc
