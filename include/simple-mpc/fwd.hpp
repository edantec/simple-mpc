///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/fwd.hpp>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc {
namespace pin = pinocchio;
using pin::FrameIndex;

using std::shared_ptr;

typedef Eigen::Matrix<double, 6, 1> eVector6;
typedef Eigen::Matrix<double, 4, 1> eVector4;
typedef Eigen::Vector3d eVector3;
typedef Eigen::Vector2d eVector2;

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
