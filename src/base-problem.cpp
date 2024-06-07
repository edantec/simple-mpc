///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/base-problem.hpp"

namespace simple_mpc {
using namespace aligator;

Problem::Problem(const Settings &settings, const RobotHandler &handler) {
  initialize(settings, handler);
}

void Problem::initialize(const Settings &settings,
                         const RobotHandler &handler) {
  settings_ = settings;
  handler_ = handler;

  nq_ = handler_.get_rmodel().nq;
  nv_ = handler_.get_rmodel().nv;
  nu_ = nv_ - 6;

  handler_.set_q0(settings_.x0.head(nq_));
}

void Problem::create_problem(std::vector<ContactMap> contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    stage_models.push_back(create_stage(cm));
  }

  problem_ = std::make_shared<TrajOptProblem>(settings_.x0, stage_models,
                                              create_terminal_cost());
}

} // namespace simple_mpc
