///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/fulldynamics.hpp"

namespace simple_mpc {
using namespace aligator;

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings &settings,
                                         const pinocchio::Model &rmodel) {
  initialize(settings, rmodel);
}

void FullDynamicsProblem::initialize(const FullDynamicsSettings &settings,
                                     const pinocchio::Model &rmodel) {
  settings_ = settings;
  rmodel_ = rmodel;

  space_ = std::make_shared<MultibodyPhaseSpace>(rmodel_);
}

} // namespace simple_mpc
