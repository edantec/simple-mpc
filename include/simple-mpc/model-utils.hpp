/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <pinocchio/multibody/fwd.hpp>

namespace pin = pinocchio;
using Model = pin::ModelTpl<double, 0>;

void makeTalosReduced(Model &model_complete, Model &model, Eigen::VectorXd &q0);
