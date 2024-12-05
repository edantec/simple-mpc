#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/workspace-base.hpp>
#include <aligator/fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/utils/exceptions.hpp>
#include <pinocchio/fwd.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fulldynamics.hpp"

namespace simple_mpc {
using namespace aligator;

FullDynamicsProblem::FullDynamicsProblem(const RobotModelHandler &model_handler, const RobotDataHandler &data_handler)
: Base(model_handler, data_handler)
{
}

FullDynamicsProblem::FullDynamicsProblem(const FullDynamicsSettings &settings, const RobotModelHandler &model_handler, const RobotDataHandler &data_handler)
: FullDynamicsProblem(model_handler, data_handler)
{

  initialize(settings);
}

void FullDynamicsProblem::initialize(const FullDynamicsSettings &settings) {

  settings_ = settings;
  actuation_matrix_.resize(nv_, nu_);
  actuation_matrix_.setZero();
  actuation_matrix_.bottomRows(nu_).setIdentity();

  prox_settings_ = ProximalSettings(1e-9, 1e-10, 1);
  x0_ = robot_model_handler_.getReferenceState();

  for (auto const &name : robot_model_handler_.getFeetNames()) {
    auto frame_ids = robot_model_handler_.getFootId(name);
    auto joint_ids = robot_model_handler_.getModel().frames[frame_ids].parentJoint;
    pinocchio::SE3 pl1 = robot_model_handler_.getModel().frames[frame_ids].placement;
    pinocchio::SE3 pl2 = robot_data_handler_.getFootPose(name);
    if (settings_.force_size == 6) {
      pinocchio::RigidConstraintModel constraint_model =
          pinocchio::RigidConstraintModel(pinocchio::ContactType::CONTACT_6D,
                                          robot_model_handler_.getModel(), joint_ids, pl1,
                                          0, pl2, pinocchio::LOCAL);
      constraint_model.corrector.Kp << 0, 0, 10, 0, 0, 0;
      constraint_model.corrector.Kd << 50, 50, 50, 50, 50, 50;
      constraint_model.name = name;
      constraint_models_.push_back(constraint_model);
    } else {
      pinocchio::RigidConstraintModel constraint_model =
          pinocchio::RigidConstraintModel(pinocchio::ContactType::CONTACT_3D,
                                          robot_model_handler_.getModel(), joint_ids, pl1,
                                          0, pl2, pinocchio::LOCAL);
      constraint_model.corrector.Kp << 0, 0, 0;
      constraint_model.corrector.Kd << 50, 50, 50;
      constraint_model.name = name;
      constraint_models_.push_back(constraint_model);
    }
  }
}

StageModel FullDynamicsProblem::createStage(
    const std::map<std::string, bool> &contact_phase,
    const std::map<std::string, pinocchio::SE3> &contact_pose,
    const std::map<std::string, Eigen::VectorXd> &contact_force,
    const std::map<std::string, bool> &land_constraint) {

  auto space = MultibodyPhaseSpace(robot_model_handler_.getModel());
  auto rcost = CostStack(space, nu_);

  rcost.addCost("state_cost",
                QuadraticStateCost(space, nu_, x0_, settings_.w_x));
  rcost.addCost(
      "control_cost",
      QuadraticControlCost(space, Eigen::VectorXd::Zero(nu_), settings_.w_u));

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, robot_model_handler_.getModel(), Eigen::VectorXd::Zero(6));
  rcost.addCost("centroidal_cost",
                QuadraticResidualCost(space, cent_mom, settings_.w_cent));

  pinocchio::context::RigidConstraintModelVector cms;

  size_t c_id = 0;
  for (auto const &name : robot_model_handler_.getFeetNames()) {
    if (settings_.force_size == 6) {
      FramePlacementResidual frame_residual = FramePlacementResidual(
          space.ndx(), nu_, robot_model_handler_.getModel(), contact_pose.at(name),
          robot_model_handler_.getFootId(name));

      rcost.addCost(
          name + "_pose_cost",
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    } else {
      FrameTranslationResidual frame_residual = FrameTranslationResidual(
          space.ndx(), nu_, robot_model_handler_.getModel(),
          contact_pose.at(name).translation(), robot_model_handler_.getFootId(name));

      rcost.addCost(
          name + "_pose_cost",
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }

    if (contact_phase.at(name))
      cms.push_back(constraint_models_[c_id]);

    c_id++;
  }

  for (auto const &name : robot_model_handler_.getFeetNames()) {
    std::shared_ptr<ContactForceResidual> frame_force;
    if (contact_force.at(name).size() != settings_.force_size) {
      throw std::runtime_error(
          "Reference forces do not have the right dimension");
    }
    if (contact_phase.at(name)) {
      frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), robot_model_handler_.getModel(), actuation_matrix_, cms,
          prox_settings_, contact_force.at(name), name);

      rcost.addCost(
          name + "_force_cost",
          QuadraticResidualCost(space, *frame_force, settings_.w_forces));
    }
  }

  MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(
      space, actuation_matrix_, cms, prox_settings_);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  StageModel stm = StageModel(rcost, dyn_model);

  // Constraints
  if (settings_.torque_limits) {
    ControlErrorResidual ctrl_fn =
        ControlErrorResidual(space.ndx(), Eigen::VectorXd::Zero(nu_));
    stm.addConstraint(ctrl_fn, BoxConstraint(settings_.umin, settings_.umax));
  }
  if (settings_.kinematics_limits) {
    StateErrorResidual state_fn =
        StateErrorResidual(space, nu_, space.neutral());
    std::vector<int> state_id;
    for (int i = 6; i < nv_; i++) {
      state_id.push_back(i);
    }
    FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
    stm.addConstraint(state_slice,
                      BoxConstraint(-settings_.qmax, -settings_.qmin));
  }

  for (auto const &name : robot_model_handler_.getFeetNames()) {
    if (settings_.force_size == 6 and contact_phase.at(name)) {
      if (settings_.force_cone) {
        MultibodyWrenchConeResidual wrench_residual =
            MultibodyWrenchConeResidual(space.ndx(), robot_model_handler_.getModel(),
                                        actuation_matrix_, cms, prox_settings_,
                                        name, settings_.mu, settings_.Lfoot,
                                        settings_.Wfoot);
        stm.addConstraint(wrench_residual, NegativeOrthant());
      }

      if (land_constraint.at(name)) {
        FrameVelocityResidual velocity_residual = FrameVelocityResidual(
            space.ndx(), nu_, robot_model_handler_.getModel(), Motion::Zero(),
            robot_model_handler_.getFootId(name), pinocchio::LOCAL);
        stm.addConstraint(velocity_residual, EqualityConstraint());
      }
    } else if (settings_.force_size == 3 and contact_phase.at(name)) {
      if (settings_.force_cone) {
        MultibodyFrictionConeResidual friction_residual =
            MultibodyFrictionConeResidual(space.ndx(), robot_model_handler_.getModel(),
                                          actuation_matrix_, cms,
                                          prox_settings_, name, settings_.mu);
        stm.addConstraint(friction_residual, NegativeOrthant());
      }
      if (land_constraint.at(name)) {
        std::vector<int> vel_id = {0, 1, 2};
        FrameVelocityResidual velocity_residual = FrameVelocityResidual(
            space.ndx(), nu_, robot_model_handler_.getModel(), Motion::Zero(),
            robot_model_handler_.getFootId(name), pinocchio::LOCAL);
        FunctionSliceXpr vel_slice =
            FunctionSliceXpr(velocity_residual, vel_id);
        stm.addConstraint(vel_slice, EqualityConstraint());

        std::vector<int> frame_id = {2};

        FrameTranslationResidual frame_residual = FrameTranslationResidual(
            space.ndx(), nu_, robot_model_handler_.getModel(),
            contact_pose.at(name).translation(), robot_model_handler_.getFootId(name));

        FunctionSliceXpr frame_slice =
            FunctionSliceXpr(frame_residual, frame_id);

        stm.addConstraint(frame_slice, EqualityConstraint());
      }
    }
  }

  return stm;
}

void FullDynamicsProblem::setReferencePoses(
    const std::size_t t,
    const std::map<std::string, pinocchio::SE3> &pose_refs) {
  if (pose_refs.size() != robot_model_handler_.getFeetNames().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }

  CostStack *cs = getCostStack(t);
  for (auto ee_name : robot_model_handler_.getFeetNames()) {
    QuadraticResidualCost *qrc =
        cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");

    if (settings_.force_size == 6) {
      FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
      cfr->setReference(pose_refs.at(ee_name));
    } else {
      FrameTranslationResidual *cfr =
          qrc->getResidual<FrameTranslationResidual>();
      cfr->setReference(pose_refs.at(ee_name).translation());
    }
  }
}

void FullDynamicsProblem::setReferencePose(const std::size_t t,
                                           const std::string &ee_name,
                                           const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getCostStack(t);
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
  if (settings_.force_size == 6) {
    FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
    cfr->setReference(pose_ref);
  } else {
    FrameTranslationResidual *cfr =
        qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref.translation());
  }
}

void FullDynamicsProblem::setTerminalReferencePose(
    const std::string &ee_name, const pinocchio::SE3 &pose_ref) {
  CostStack *cs = getTerminalCostStack();
  QuadraticResidualCost *qrc =
      cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
  if (settings_.force_size == 6) {
    FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
    cfr->setReference(pose_ref);
  } else {
    FrameTranslationResidual *cfr =
        qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref.translation());
  }
}

void FullDynamicsProblem::setReferenceForces(
    const std::size_t t,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  CostStack *cs = getCostStack(t);
  if (force_refs.size() != robot_model_handler_.getFeetNames().size()) {
    throw std::runtime_error(
        "force_refs size does not match number of end effectors");
  }
  for (auto ee_name : robot_model_handler_.getFeetNames()) {
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
  if (settings_.force_size == 6) {
    FramePlacementResidual *cfr = qrc->getResidual<FramePlacementResidual>();
    return cfr->getReference();
  } else {
    FrameTranslationResidual *cfr =
        qrc->getResidual<FrameTranslationResidual>();
    SE3 ref = SE3::Identity();
    ref.translation() = cfr->getReference();
    return ref;
  }
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

const Eigen::VectorXd
FullDynamicsProblem::getVelocityBase(const std::size_t t) {
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  return qc->getTarget().segment(nq_, 6);
}

void FullDynamicsProblem::setVelocityBase(
    const std::size_t t, const Eigen::VectorXd &velocity_base) {
  if (velocity_base.size() != 6) {
    throw std::runtime_error("velocity_base size should be 6");
  }
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  x0_.segment(nq_, 6) = velocity_base;
  qc->setTarget(x0_);
}

const Eigen::VectorXd FullDynamicsProblem::getPoseBase(const std::size_t t) {
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  return qc->getTarget().head(7);
};

void FullDynamicsProblem::setPoseBase(const std::size_t t,
                                      const Eigen::VectorXd &pose_base) {
  if (pose_base.size() != 7) {
    throw std::runtime_error("pose_base size should be 7");
  }
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  x0_.head(7) = pose_base;
  qc->setTarget(x0_);
}

const Eigen::VectorXd FullDynamicsProblem::getProblemState() {
  return x0_;
}

size_t FullDynamicsProblem::getContactSupport(const std::size_t t) {
  MultibodyConstraintFwdDynamics *ode =
      problem_->stages_[t]
          ->getDynamics<IntegratorSemiImplEuler>()
          ->getDynamics<MultibodyConstraintFwdDynamics>();

  return ode->constraint_models_.size();
}

CostStack FullDynamicsProblem::createTerminalCost() {
  auto ter_space = MultibodyPhaseSpace(robot_model_handler_.getModel());
  auto term_cost = CostStack(ter_space, nu_);
  auto cent_mom = CentroidalMomentumResidual(
      ter_space.ndx(), nu_, robot_model_handler_.getModel(), Eigen::VectorXd::Zero(6));

  term_cost.addCost("state_cost",
                    QuadraticStateCost(ter_space, nu_, x0_, settings_.w_x));
  /* term_cost.addCost(
      "centroidal_cost",
      QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent)); */

  return term_cost;
}

void FullDynamicsProblem::createTerminalConstraint() {
  if (!problem_initialized_) {
    throw std::runtime_error("Create problem first!");
  }
  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ndx_, nu_, robot_model_handler_.getModel(), robot_data_handler_.getData().com[0]);

  double tau = sqrt(robot_data_handler_.getData().com[0][2] / 9.81);
  DCMPositionResidual dcm_cstr = DCMPositionResidual(
      ndx_, nu_, robot_model_handler_.getModel(), robot_data_handler_.getData().com[0], tau);

  problem_->addTerminalConstraint(dcm_cstr, EqualityConstraint());

  Motion v_ref = Motion::Zero();
  for (auto const &name : robot_model_handler_.getFeetNames()) {
    FrameVelocityResidual frame_vel =
        FrameVelocityResidual(ndx_, nu_, robot_model_handler_.getModel(), v_ref,
                              robot_model_handler_.getFootId(name), pinocchio::LOCAL);
    if (settings_.force_size == 6)
      problem_->addTerminalConstraint(frame_vel, EqualityConstraint());
    else {
      std::vector<int> vel_id = {0, 1, 2};

      FunctionSliceXpr vel_slice = FunctionSliceXpr(frame_vel, vel_id);
      problem_->addTerminalConstraint(vel_slice, EqualityConstraint());
    }
  }

  terminal_constraint_ = true;
}

void FullDynamicsProblem::updateTerminalConstraint(
    const Eigen::Vector3d &com_ref) {
  if (terminal_constraint_) {
    DCMPositionResidual *CoMres =
        problem_->term_cstrs_.getConstraint<DCMPositionResidual>(0);

    CoMres->setReference(com_ref);
  }
}

} // namespace simple_mpc
