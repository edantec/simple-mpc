#include "simple-mpc/kinodynamics.hpp"

namespace simple_mpc {
using namespace aligator;

KinodynamicsProblem::KinodynamicsProblem(const RobotHandler &handler)
    : Base(handler) {}

KinodynamicsProblem::KinodynamicsProblem(const KinodynamicsSettings &settings,
                                         const RobotHandler &handler)
    : Base(handler) {
  initialize(settings);
}

void KinodynamicsProblem::initialize(const KinodynamicsSettings &settings) {
  settings_ = settings;
  nu_ = nv_ - 6 + settings_.force_size * (int)handler_.getFeetNames().size();
  x0_ = handler_.getState();
  control_ref_.resize(nu_);
  control_ref_.setZero();
}

StageModel KinodynamicsProblem::createStage(
    const std::map<std::string, bool> &contact_phase,
    const std::map<std::string, pinocchio::SE3> &contact_pose,
    const std::map<std::string, Eigen::VectorXd> &contact_force,
    const std::map<std::string, bool> &land_constraint) {
  auto space = MultibodyPhaseSpace(handler_.getModel());
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states;
  for (auto const &x : contact_phase) {
    contact_states.push_back(x.second);
  }

  computeControlFromForces(contact_force);

  auto cent_mom = CentroidalMomentumResidual(
      space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));
  auto centder_mom = CentroidalMomentumDerivativeResidual(
      space.ndx(), handler_.getModel(), settings_.gravity, contact_states,
      handler_.getFeetIds(), settings_.force_size);
  rcost.addCost("state_cost",
                QuadraticStateCost(space, nu_, x0_, settings_.w_x));
  rcost.addCost("control_cost",
                QuadraticControlCost(space, control_ref_, settings_.w_u));
  rcost.addCost("centroidal_cost",
                QuadraticResidualCost(space, cent_mom, settings_.w_cent));
  rcost.addCost("centroidal_derivative_cost",
                QuadraticResidualCost(space, centder_mom, settings_.w_centder));

  for (auto const &name : handler_.getFeetNames()) {
    if (settings_.force_size == 6) {
      FramePlacementResidual frame_residual = FramePlacementResidual(
          space.ndx(), nu_, handler_.getModel(), contact_pose.at(name),
          handler_.getFootId(name));

      rcost.addCost(
          name + "_pose_cost",
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    } else {
      FrameTranslationResidual frame_residual = FrameTranslationResidual(
          space.ndx(), nu_, handler_.getModel(),
          contact_pose.at(name).translation(), handler_.getFootId(name));

      rcost.addCost(
          name + "_pose_cost",
          QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
  }

  KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, handler_.getModel(), settings_.gravity, contact_states,
      handler_.getFeetIds(), settings_.force_size);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);
  StageModel stm = StageModel(rcost, dyn_model);

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

  Motion v_ref = Motion::Zero();
  int i = 0;
  for (auto const &name : handler_.getFeetNames()) {
    if (contact_phase.at(name)) {
      FrameVelocityResidual frame_vel =
          FrameVelocityResidual(space.ndx(), nu_, handler_.getModel(), v_ref,
                                handler_.getFootId(name), pinocchio::LOCAL);
      if (settings_.force_size == 6) {
        if (settings_.force_cone) {
          CentroidalWrenchConeResidual wrench_residual =
              CentroidalWrenchConeResidual(space.ndx(), nu_, i, settings_.mu,
                                           settings_.Lfoot, settings_.Wfoot);
          stm.addConstraint(wrench_residual, NegativeOrthant());
        }
        stm.addConstraint(frame_vel, EqualityConstraint());
      } else {
        if (settings_.force_cone) {
          CentroidalFrictionConeResidual friction_residual =
              CentroidalFrictionConeResidual(space.ndx(), nu_, i, settings_.mu,
                                             1e-4);
          stm.addConstraint(friction_residual, NegativeOrthant());
        }
        std::vector<int> vel_id = {0, 1, 2};

        FunctionSliceXpr vel_slice = FunctionSliceXpr(frame_vel, vel_id);
        stm.addConstraint(vel_slice, EqualityConstraint());
        if (land_constraint.at(name)) {
          std::vector<int> frame_id = {2};

          FrameTranslationResidual frame_residual = FrameTranslationResidual(
              space.ndx(), nu_, handler_.getModel(),
              contact_pose.at(name).translation(), handler_.getFootId(name));

          FunctionSliceXpr frame_slice =
              FunctionSliceXpr(frame_residual, frame_id);

          stm.addConstraint(frame_slice, EqualityConstraint());
        }
      }
    }
    i++;
  }

  return stm;
}

void KinodynamicsProblem::setReferencePose(const std::size_t t,
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

void KinodynamicsProblem::setReferencePoses(
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

void KinodynamicsProblem::setTerminalReferencePose(
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

const pinocchio::SE3
KinodynamicsProblem::getReferencePose(const std::size_t t,
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

void KinodynamicsProblem::computeControlFromForces(
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  for (std::size_t i = 0; i < handler_.getFeetNames().size(); i++) {
    if (settings_.force_size != force_refs.at(handler_.getFootName(i)).size()) {
      throw std::runtime_error(
          "force size in settings does not match reference force size");
    }
    control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(handler_.getFootName(i));
  }
}

void KinodynamicsProblem::setReferenceForces(
    const std::size_t i,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  computeControlFromForces(force_refs);
  setReferenceControl(i, control_ref_);
}

void KinodynamicsProblem::setReferenceForce(const std::size_t i,
                                            const std::string &ee_name,
                                            const Eigen::VectorXd &force_ref) {
  std::vector<std::string> hname = handler_.getFeetNames();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();
  control_ref_.segment(id * settings_.force_size, settings_.force_size) =
      force_ref;
  setReferenceControl(i, control_ref_);
}

const Eigen::VectorXd
KinodynamicsProblem::getReferenceForce(const std::size_t i,
                                       const std::string &ee_name) {
  std::vector<std::string> hname = handler_.getFeetNames();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();

  return getReferenceControl(i).segment(id * settings_.force_size,
                                        settings_.force_size);
}

const Eigen::VectorXd
KinodynamicsProblem::getVelocityBase(const std::size_t t) {
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  return qc->getTarget().segment(nq_, 6);
}

void KinodynamicsProblem::setVelocityBase(
    const std::size_t t, const Eigen::VectorXd &velocity_base) {
  CostStack *cs = getCostStack(t);
  QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
  x0_.segment(nq_, 6) = velocity_base;
  qc->setTarget(x0_);
}

const Eigen::VectorXd KinodynamicsProblem::getProblemState() {
  return handler_.getState();
}

size_t KinodynamicsProblem::getContactSupport(const std::size_t t) {
  KinodynamicsFwdDynamics *ode = problem_->stages_[t]
                                     ->getDynamics<IntegratorSemiImplEuler>()
                                     ->getDynamics<KinodynamicsFwdDynamics>();

  size_t active_contacts = 0;
  for (auto const contact : ode->contact_states_) {
    if (contact) {
      active_contacts += 1;
    }
  }
  return active_contacts;
}

CostStack KinodynamicsProblem::createTerminalCost() {
  auto ter_space = MultibodyPhaseSpace(handler_.getModel());
  auto term_cost = CostStack(ter_space, nu_);
  auto cent_mom = CentroidalMomentumResidual(
      ter_space.ndx(), nu_, handler_.getModel(), Eigen::VectorXd::Zero(6));

  term_cost.addCost("state_cost",
                    QuadraticStateCost(ter_space, nu_, x0_, settings_.w_x));
  term_cost.addCost(
      "centroidal_cost",
      QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent * 10));

  return term_cost;
}

void KinodynamicsProblem::createTerminalConstraint() {
  if (!problem_initialized_) {
    throw std::runtime_error("Create problem first!");
  }
  CenterOfMassTranslationResidual com_cstr = CenterOfMassTranslationResidual(
      ndx_, nu_, handler_.getModel(), handler_.getComPosition());

  problem_->addTerminalConstraint(com_cstr, EqualityConstraint());
  terminal_constraint_ = true;
}

void KinodynamicsProblem::updateTerminalConstraint(
    const Eigen::Vector3d &com_ref) {
  if (terminal_constraint_) {
    CenterOfMassTranslationResidual *CoMres =
        problem_->term_cstrs_.getConstraint<CenterOfMassTranslationResidual>(0);

    CoMres->setReference(com_ref);
  }
}

} // namespace simple_mpc
