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

void Problem::create_problem(const Eigen::VectorXd &x0,
                             const std::vector<ContactMap> &contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    std::vector<bool> contact_states = cm.getContactStates();
    std::vector<Eigen::VectorXd> force_ref;
    for (std::size_t i = 0; i < contact_states.size(); i++) {
      force_ref.push_back(Eigen::VectorXd::Zero(6));
    }
    stage_models.push_back(create_stage(cm, force_ref));
  }

  problem_ = std::make_shared<TrajOptProblem>(x0, stage_models,
                                              create_terminal_cost());
}

StageModel
Problem::create_stage(const ContactMap &contact_map,
                      const std::vector<Eigen::VectorXd> &force_refs) {
  auto space = VectorSpace(10);
  auto rcost = CostStack(space, nu_);
  Eigen::Vector3d gravity;
  gravity << 0, 0, 9;

  CentroidalFwdDynamics ode =
      CentroidalFwdDynamics(space, handler_.get_mass(), gravity, contact_map,
                            (int)force_refs[0].size());
  IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, 0.01);

  return StageModel(rcost, dyn_model);
}

CostStack Problem::create_terminal_cost() {
  auto ter_space = VectorSpace(10);
  auto term_cost = CostStack(ter_space, nu_);

  return term_cost;
}

void Problem::set_reference_control(const std::size_t i,
                                    const Eigen::VectorXd &u_ref) {
  if (i >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[i]->cost_);
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

} // namespace simple_mpc
