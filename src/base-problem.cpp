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
    const std::vector<std::map<std::string, bool>> &contact_phases,
    const std::vector<std::map<std::string, pinocchio::SE3>> &contact_poses,
    const std::vector<std::map<std::string, Eigen::VectorXd>> &contact_forces) {
  if (contact_phases.size() != contact_poses.size()) {
    throw std::runtime_error(
        "Contact phases and poses sequences do not have the same size");
  }
  if (contact_phases.size() != contact_forces.size()) {
    throw std::runtime_error(
        "Contact phases and forces sequences do not have the same size");
  }
  std::map<std::string, bool> previous_phases;
  for (auto const &name : handler_.settings_.getFeetNames()) {
    previous_phases.insert({name, true});
  }
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (std::size_t i = 0; i < contact_phases.size(); i++) {
    std::map<std::string, bool> land_constraint;
    for (auto const &name : handler_.settings_getFeetNames()) {
      if (!previous_phases.at(name) and contact_phases[i].at(name))
        land_constraint.insert({name, true});
      else
        land_constraint.insert({name, false});
    }
    StageModel stage = createStage(contact_phases[i], contact_poses[i],
                                   contact_forces[i], land_constraint);
    stage_models.push_back(stage);
    previous_phases = contact_phases[i];
  }

  return stage_models;
}

void Problem::setReferenceControl(const std::size_t t,
                                  const Eigen::VectorXd &u_ref) {
  CostStack *cs = getCostStack(t);
  QuadraticControlCost *qc =
      cs->getComponent<QuadraticControlCost>("control_cost");
  qc->setTarget(u_ref);
}

const Eigen::VectorXd Problem::getReferenceControl(const std::size_t t) {
  CostStack *cs = getCostStack(t);
  QuadraticControlCost *qc =
      cs->getComponent<QuadraticControlCost>("control_cost");
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
  std::vector<std::map<std::string, bool>> contact_phases;
  std::vector<std::map<std::string, pinocchio::SE3>> contact_poses;
  std::vector<std::map<std::string, Eigen::VectorXd>> contact_forces;

  Eigen::VectorXd force_ref(force_size);
  force_ref.setZero();
  force_ref[2] =
      -handler_.getMass() * gravity / (double)handler_.settings_.getFeetNames().size();

  std::map<std::string, bool> contact_phase;
  std::map<std::string, pinocchio::SE3> contact_pose;
  std::map<std::string, Eigen::VectorXd> contact_force;
  for (auto &name : handler_.settings_.getFeetNames()) {
    contact_phase.insert({name, true});
    contact_pose.insert({name, handler_.getFootPose(name)});
    contact_force.insert({name, force_ref});
  }

  for (size_t i = 0; i < horizon; i++) {
    contact_phases.push_back(contact_phase);
    contact_poses.push_back(contact_pose);
    contact_forces.push_back(contact_force);
  }
  std::vector<xyz::polymorphic<StageModel>> stage_models =
      createStages(contact_phases, contact_poses, contact_forces);

  problem_ =
      std::make_shared<TrajOptProblem>(x0, stage_models, createTerminalCost());
  problem_initialized_ = true;

  // createTerminalConstraint();
}
} // namespace simple_mpc
