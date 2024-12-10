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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using VectorRef = Eigen::Ref<VectorXd>;
using ConstVectorRef = Eigen::Ref<const VectorXd>;

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;

} // namespace simple_mpc
