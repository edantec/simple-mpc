///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/base-problem.hpp"
#include <stdexcept>

namespace simple_mpc {
using namespace aligator;

Problem::~Problem() {}

Problem::Problem(const RobotHandler &handler) : handler_(handler) {
  nq_ = handler_.get_rmodel().nq;
  nv_ = handler_.get_rmodel().nv;
  nu_ = nv_ - 6;
}

std::vector<xyz::polymorphic<StageModel>> Problem::create_stages(
    const std::vector<ContactMap> &contact_sequence,
    const std::vector<std::map<std::string, Eigen::VectorXd>> &force_sequence) {
  if (contact_sequence.size() != force_sequence.size()) {
    throw std::runtime_error(
        "Contact and force sequences do not have the same size");
  }
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (std::size_t i = 0; i < contact_sequence.size(); i++) {
    stage_models.push_back(
        create_stage(contact_sequence[i], force_sequence[i]));
  }

  return stage_models;
}

void Problem::set_reference_control(const std::size_t i,
                                    const Eigen::VectorXd &u_ref) {
  CostStack *cs = get_cost_stack(i);
  QuadraticControlCost *qc = dynamic_cast<QuadraticControlCost *>(
      &*cs->components_[cost_map_.at("control_cost")]);

  qc->setTarget(u_ref);
}

Eigen::VectorXd Problem::get_reference_control(const std::size_t t) {
  CostStack *cs = get_cost_stack(t);
  QuadraticControlCost *qc = dynamic_cast<QuadraticControlCost *>(
      &*cs->components_[cost_map_.at("control_cost")]);

  return qc->getTarget();
}

CostStack *Problem::get_cost_stack(std::size_t t) {
  if (t >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

  return cs;
}

std::size_t Problem::get_cost_number() {
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[0]->cost_);
  return cs->components_.size();
}

std::size_t Problem::get_size() { return problem_->stages_.size(); }

} // namespace simple_mpc
