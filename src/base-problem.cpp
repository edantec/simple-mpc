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

Problem::~Problem() {}

Problem::Problem(const RobotHandler &handler) : handler_(handler) {
  nq_ = handler_.get_rmodel().nq;
  nv_ = handler_.get_rmodel().nv;
  nu_ = nv_ - 6;
}

std::vector<xyz::polymorphic<StageModel>>
Problem::create_stages(const std::vector<ContactMap> &contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    std::vector<bool> contact_states = cm.getContactStates();
    std::vector<Eigen::VectorXd> force_ref;
    for (std::size_t i = 0; i < contact_states.size(); i++) {
      force_ref.push_back(Eigen::VectorXd::Zero(6));
    }
    stage_models.push_back(create_stage(cm, force_ref));
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

void Problem::insert_cost(CostStack &cost_stack,
                          const xyz::polymorphic<CostAbstract> &cost,
                          std::map<std::string, std::size_t> &cost_map,
                          const std::string &name, int &cost_incr) {
  cost_stack.addCost(cost);
  cost_map.insert({name, cost_incr});
  cost_incr++;
}

CostStack *Problem::get_cost_stack(std::size_t i) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);

  return cs;
}

std::size_t Problem::get_cost_number() {
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[0]->cost_);
  return cs->components_.size();
}

} // namespace simple_mpc
