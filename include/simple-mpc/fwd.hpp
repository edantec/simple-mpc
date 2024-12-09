///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/fwd.hpp>
#include <aligator/modelling/contact-map.hpp>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc {
namespace pin = pinocchio;
using pin::FrameIndex;
using std::shared_ptr;
using ContactMap = aligator::ContactMapTpl<double>;

// MPC
struct Settings;
struct FullDynamicsSettings;
class RobotHandler;
class FullDynamicsProblem;
class KinodynamicsProblem;
class CentroidalProblem;
class OCPHandler;
class IDSolver;
class IKIDSolver;

} // namespace simple_mpc
