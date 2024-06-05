/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "aligator/core/traj-opt-problem.hpp"

#include <Eigen/Core>

#include <pinocchio/multibody/fwd.hpp>

namespace pin = pinocchio;

using aligator::context::TrajOptData;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Model = pin::ModelTpl<double, 0>;

void makeTalosReduced(Model &model_complete, Model &model, Eigen::VectorXd &q0);
