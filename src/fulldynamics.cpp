#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/utils/exceptions.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/fulldynamics.hpp"

namespace simple_mpc {
using namespace aligator;

FullDynamicsProblem::FullDynamicsProblem(const RobotHandler &handler)
    : Base(handler) {}

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings &settings,
                                         const RobotHandler &handler)
    : Base(handler) {

  initialize(settings);
}

void FullDynamicsProblem::initialize(const FullDynamicsSettings &settings) {

  settings_ = settings;
  actuation_matrix_.resize(nv_, nu_);
  actuation_matrix_.setZero();
  actuation_matrix_.bottomRows(nu_).setIdentity();

  prox_settings_ = ProximalSettings(1e-9, 1e-10, 1);

  for (auto const &name : handler_.getFeetNames()) {
    auto frame_ids = handler_.getFootId(name);
    auto joint_ids = handler_.getModel().frames[frame_ids].parentJoint;
    pinocchio::SE3 pl1 = handler_.getModel().frames[frame_ids].placement;
    pinocchio::SE3 pl2 = handler_.getFootPose(name);
    pinocchio::RigidConstraintModel constraint_model =
        pinocchio::RigidConstraintModel(pinocchio::ContactType::CONTACT_6D,
                                        handler_.getModel(), joint_ids, pl1, 0,
                                        pl2, pinocchio::LOCAL);
    constraint_model.corrector.Kp << 1, 1, 10, 1, 1, 1;
    constraint_model.corrector.Kd << 50, 50, 50, 50, 50, 50;
    constraint_model.name = name;
    constraint_models_.push_back(constraint_model);
  }
}

StageModel FullDynamicsProblem::createStage(
    const std::map<std::string, bool> &contact_phase,
    const std::map<std::string, pinocchio::SE3> &contact_pose,
    const std::map<std::string, Eigen::VectorXd> &contact_force) {

  auto space = MultibodyPhaseSpace(handler_.getModel());
  auto rcost = CostStack(space, nu_);

  rcost.addCost("state_cost",
                QuadraticStateCost(space, nu_, settings_.x0, settings_.w_x));
  rcost.addCost("control_cost",
                QuadraticControlCost(space, settings_.u0, settings_.w_u));

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));
  rcost.addCost("centroidal_cost",
                QuadraticResidualCost(space, cent_mom, settings_.w_cent));

  pinocchio::context::RigidConstraintModelVector cms;

  size_t c_id = 0;
  for (auto const &name : handler_.getFeetNames()) {
    FramePlacementResidual frame_residual =
        FramePlacementResidual(space.ndx(), nu_, handler_.getModel(),
                               contact_pose.at(name), handler_.getFootId(name));

    if (contact_phase.at(name))
      cms.push_back(constraint_models_[c_id]);

    rcost.addCost(
        name + "_pose_cost",
        QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    c_id++;
  }

  for (auto const &name : handler_.getFeetNames()) {
    std::shared_ptr<ContactForceResidual> frame_force;
    int is_active = 0;
    if (contact_phase.at(name)) {
      frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), handler_.getModel(), actuation_matrix_, cms,
          prox_settings_, contact_force.at(name), name);
      is_active = 1;
    } else {
      frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), handler_.getModel(), actuation_matrix_,
          constraint_models_, prox_settings_, contact_force.at(name), name);
    }
    rcost.addCost(name + "_force_cost",
                  QuadraticResidualCost(space, *frame_force,
                                        settings_.w_forces * is_active));
  }

  MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(
      space, actuation_matrix_, cms, prox_settings_);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  StageModel stm = StageModel(rcost, dyn_model);

  // Constraints
  ControlErrorResidual ctrl_fn =
      ControlErrorResidual(space.ndx(), Eigen::VectorXd::Zero(nu_));
  stm.addConstraint(ctrl_fn, BoxConstraint(settings_.umin, settings_.umax));
  StateErrorResidual state_fn = StateErrorResidual(space, nu_, space.neutral());
  std::vector<int> state_id;
  for (int i = 6; i < nv_; i++) {
    state_id.push_back(i);
  }
  FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
  stm.addConstraint(state_slice,
                    BoxConstraint(-settings_.qmax, -settings_.qmin));

  for (auto const &name : handler_.getFeetNames()) {
    if (contact_phase.at(name)) {
      MultibodyWrenchConeResidual wrench_residual = MultibodyWrenchConeResidual(
          space.ndx(), handler_.getModel(), actuation_matrix_, cms,
          prox_settings_, name, settings_.mu, settings_.Lfoot, settings_.Wfoot);
      stm.addConstraint(wrench_residual, NegativeOrthant());
    }
  }

  return stm;
}

void FullDynamicsProblem::setReferencePoses(
    const std::size_t t,
    const std::map<std::string, pinocchio::SE3> &pose_refs) {
  if (pose_refs.size() != handler_.getFeetNames().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }

  CostStack *cs = getCostStack(t);
  for (auto ee_name : handler_.getFeetNames()) {
    QuadraticResidualCost *qrc =
        cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
    cfr->setReference(pose_refs.at(ee_name));
  }
}

void FullDynamicsProblem::setReferencePose(const std::size_t t,
                                           const std::string &ee_name,
                                           const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
  FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
  cfr->setReference(pose_ref);
}

void FullDynamicsProblem::setTerminalReferencePose(
    const std::string &ee_name, const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getTerminalCostStack();
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
  FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
  cfr->setReference(pose_ref);
}

void FullDynamicsProblem::setReferenceForces(
    const std::size_t t,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  CostStack *cs = getCostStack(t);
  if (force_refs.size() != handler_.getFeetNames().size()) {
    throw std::runtime_error(
        "force_refs size does not match number of end effectors");
  }
  for (auto ee_name : handler_.getFeetNames()) {
    QuadraticResidualCost *qrc =
        cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
    ContactForceResidual *cfr = qrc->getResidual<ContactForceResidual>();
    cfr->setReference(force_refs.at(ee_name));
  }
}

void FullDynamicsProblem::setReferenceForce(const std::size_t i,
                                            const std::string &ee_name,
                                            const Eigen::VectorXd &force_ref) {
  CostStack *cs = getCostStack(i);
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
  ContactForceResidual *cfr = qrc->getResidual<ContactForceResidual>();
  cfr->setReference(force_ref);
}

const pinocchio::SE3
FullDynamicsProblem::getReferencePose(const std::size_t t,
                                      const std::string &ee_name) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
  FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
  return cfr->getReference();
}

const Eigen::VectorXd
FullDynamicsProblem::getReferenceForce(const std::size_t t,
                                       const std::string &ee_name) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
  ContactForceResidual *cfr = qrc->getResidual<ContactForceResidual>();
  return cfr->getReference();
}

const Eigen::VectorXd FullDynamicsProblem::getProblemState() {
  return handler_.getState();
}

CostStack FullDynamicsProblem::createTerminalCost() {
  auto ter_space = MultibodyPhaseSpace(handler_.getModel());
  auto term_cost = CostStack(ter_space, nu_);
  auto cent_mom = CentroidalMomentumResidual(
      ter_space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));

  term_cost.addCost(
      "state_cost",
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));
  term_cost.addCost("control_cost", QuadraticResidualCost(ter_space, cent_mom,
                                                          settings_.w_cent));
  for (auto const &name : handler_.getFeetNames()) {
    FramePlacementResidual frame_residual = FramePlacementResidual(
        ter_space.ndx(), nu_, handler_.getModel(), handler_.getFootPose(name),
        handler_.getFootId(name));

    term_cost.addCost(
        name + "_pose_cost",
        QuadraticResidualCost(ter_space, frame_residual, settings_.w_frame));
  }

  return term_cost;
}

void FullDynamicsProblem::createTerminalConstraint() {
  if (!problem_initialized_) {
    throw std::runtime_error("Create problem first!");
  }
  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ndx_, nu_, handler_.getModel(), handler_.getComPosition());

  StageConstraint term_constraint_com = {com_cstr, EqualityConstraint()};
  problem_->addTerminalConstraint(term_constraint_com);

  terminal_constraint_ = true;
}

void FullDynamicsProblem::updateTerminalConstraint(
    const Eigen::Vector3d &com_ref) {
  if (terminal_constraint_) {
    CenterOfMassTranslationResidual *CoMres =
        problem_->term_cstrs_.getConstraint<CenterOfMassTranslationResidual>(0);

    CoMres->setReference(com_ref);
  }
}

} // namespace simple_mpc
