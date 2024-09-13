#include "simple-mpc/base-problem.hpp"
#include <stdexcept>

namespace simple_mpc {
using namespace aligator;

Problem::~Problem() {}

Problem::Problem(const RobotHandler &handler) : handler_(handler) {
  nq_ = handler_.getModel().nq;
  nv_ = handler_.getModel().nv;
  ndx_ = 2 * handler_.getModel().nv;
  nu_ = nv_ - 6;
}

std::vector<xyz::polymorphic<StageModel>> Problem::createStages(
    const std::vector<ContactMap> &contact_sequence,
    const std::vector<std::map<std::string, Eigen::VectorXd>> &force_sequence) {
  if (contact_sequence.size() != force_sequence.size()) {
    throw std::runtime_error(
        "Contact and force sequences do not have the same size");
  }
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (std::size_t i = 0; i < contact_sequence.size(); i++) {
    stage_models.push_back(createStage(contact_sequence[i], force_sequence[i]));
  }

  return stage_models;
}

void Problem::setReferenceControl(const std::size_t i,
                                  const Eigen::VectorXd &u_ref) {
  CostStack *cs = getCostStack(i);
  QuadraticControlCost *qc = dynamic_cast<QuadraticControlCost *>(
      &*cs->components_[cost_map_.at("control_cost")]);

  qc->setTarget(u_ref);
}

const Eigen::VectorXd Problem::getReferenceControl(const std::size_t t) {
  CostStack *cs = getCostStack(t);
  QuadraticControlCost *qc = dynamic_cast<QuadraticControlCost *>(
      &*cs->components_[cost_map_.at("control_cost")]);

  return qc->getTarget();
}

CostStack *Problem::getCostStack(std::size_t t) {
  if (t >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

  return cs;
}

CostStack *Problem::getTerminalCostStack() {
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->term_cost_);

  return cs;
}

std::size_t Problem::getCostNumber() {
  CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[0]->cost_);
  return cs->components_.size();
}

std::size_t Problem::getSize() { return problem_->stages_.size(); }

void Problem::createProblem(const Eigen::VectorXd &x0, const size_t horizon,
                            const int force_size, const double gravity) {

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;

  Eigen::VectorXd force_ref(force_size);
  force_ref.setZero();
  force_ref[2] =
      -handler_.getMass() * gravity / (double)handler_.getFeetNames().size();

  std::vector<bool> contact_states;
  aligator::StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  std::map<std::string, Eigen::VectorXd> force_map;
  for (auto &name : handler_.getFeetNames()) {
    contact_states.push_back(true);
    contact_poses.push_back(handler_.getFootPose(name).translation());
    force_map.insert({name, force_ref});
  }

  for (size_t i = 0; i < horizon; i++) {
    ContactMap contact_map(handler_.getFeetNames(), contact_states,
                           contact_poses);
    contact_sequence.push_back(contact_map);
    force_sequence.push_back(force_map);
  }
  std::vector<xyz::polymorphic<StageModel>> stage_models =
      createStages(contact_sequence, force_sequence);

  problem_ =
      std::make_shared<TrajOptProblem>(x0, stage_models, createTerminalCost());

  problem_initialized_ = true;

  // createTerminalConstraint();
}
} // namespace simple_mpc
