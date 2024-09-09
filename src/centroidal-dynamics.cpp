#include "simple-mpc/centroidal-dynamics.hpp"
#include <pinocchio/spatial/fwd.hpp>
#include <stdexcept>

namespace simple_mpc {
using namespace aligator;

CentroidalProblem::CentroidalProblem(const RobotHandler &handler)
    : Base(handler) {}

CentroidalProblem::CentroidalProblem(const CentroidalSettings &settings,
                                     const RobotHandler &handler)
    : Base(handler) {
  initialize(settings);
}

void CentroidalProblem::initialize(const CentroidalSettings &settings) {
  settings_ = settings;
  nx_ = 9;
  nu_ = (int)handler_.get_ee_names().size() * settings_.force_size;
  if (nu_ != settings_.u0.size()) {
    throw std::runtime_error("settings.u0 does not have the correct size nu");
  }
  control_ref_ = settings_.u0;

  // Set up cost names used in kinodynamics problem
  std::size_t cost_incr = 0;
  cost_map_.insert({"control_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"linear_mom_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"angular_mom_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"linear_acc_cost", cost_incr});
  cost_incr++;
  cost_map_.insert({"angular_acc_cost", cost_incr});
  cost_incr++;
}

StageModel CentroidalProblem::create_stage(
    const ContactMap &contact_map,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  auto space = VectorSpace(nx_);
  auto rcost = CostStack(space, nu_);
  std::vector<bool> contact_states = contact_map.getContactStates();
  auto contact_poses = contact_map.getContactPoses();

  compute_control_from_forces(force_refs);

  auto linear_mom = LinearMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
  auto angular_mom = AngularMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());

  auto linear_acc = CentroidalAccelerationResidual(
      space.ndx(), nu_, handler_.get_mass(), settings_.gravity, contact_map,
      settings_.force_size);
  auto angular_acc = AngularAccelerationResidual(
      space.ndx(), nu_, handler_.get_mass(), settings_.gravity, contact_map,
      settings_.force_size);

  rcost.addCost(QuadraticControlCost(space, control_ref_, settings_.w_u));
  rcost.addCost(
      QuadraticResidualCost(space, linear_mom, settings_.w_linear_mom));
  rcost.addCost(
      QuadraticResidualCost(space, angular_mom, settings_.w_angular_mom));
  rcost.addCost(
      QuadraticResidualCost(space, linear_acc, settings_.w_linear_acc));
  rcost.addCost(
      QuadraticResidualCost(space, angular_acc, settings_.w_angular_acc));

  CentroidalFwdDynamics ode =
      CentroidalFwdDynamics(space, handler_.get_mass(), settings_.gravity,
                            contact_map, settings_.force_size);
  IntegratorSemiImplEuler dyn_model =
      IntegratorSemiImplEuler(ode, settings_.DT);

  return StageModel(rcost, dyn_model);
}

void CentroidalProblem::compute_control_from_forces(
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  for (std::size_t i = 0; i < handler_.get_ee_names().size(); i++) {
    if (settings_.force_size != force_refs.at(handler_.get_ee_name(i)).size()) {
      throw std::runtime_error(
          "force size in settings does not match reference force size");
    }
    control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(handler_.get_ee_name(i));
  }
}

void CentroidalProblem::set_reference_poses(
    const std::size_t t,
    const std::map<std::string, pinocchio::SE3> &pose_refs) {
  if (t >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  if (pose_refs.size() != handler_.get_ee_names().size()) {
    throw std::runtime_error(
        "pose_refs size does not match number of end effectors");
  }
  IntegratorSemiImplEuler *dyn = dynamic_cast<IntegratorSemiImplEuler *>(
      &*problem_->stages_[t]->dynamics_);
  CentroidalFwdDynamics *cent_dyn =
      dynamic_cast<CentroidalFwdDynamics *>(&*dyn->ode_);

  for (auto const &pose : pose_refs) {
    cent_dyn->contact_map_.setContactPose(pose.first,
                                          pose.second.translation());
  }
  CostStack *cs = get_cost_stack(t);

  for (auto ee_name : handler_.get_ee_names()) {
    QuadraticResidualCost *qrc1 = dynamic_cast<QuadraticResidualCost *>(
        &*cs->components_[cost_map_.at("linear_acc_cost")]);
    QuadraticResidualCost *qrc2 = dynamic_cast<QuadraticResidualCost *>(
        &*cs->components_[cost_map_.at("angular_acc_cost")]);
    CentroidalAccelerationResidual *car =
        dynamic_cast<CentroidalAccelerationResidual *>(&*qrc1->residual_);
    AngularAccelerationResidual *aar =
        dynamic_cast<AngularAccelerationResidual *>(&*qrc2->residual_);
    car->contact_map_.setContactPose(ee_name,
                                     pose_refs.at(ee_name).translation());
    aar->contact_map_.setContactPose(ee_name,
                                     pose_refs.at(ee_name).translation());
  }
}

void CentroidalProblem::set_reference_pose(const std::size_t t,
                                           const std::string &ee_name,
                                           const pinocchio::SE3 &pose_ref) {
  if (t >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  IntegratorSemiImplEuler *dyn = dynamic_cast<IntegratorSemiImplEuler *>(
      &*problem_->stages_[t]->dynamics_);
  CentroidalFwdDynamics *cent_dyn =
      dynamic_cast<CentroidalFwdDynamics *>(&*dyn->ode_);
  cent_dyn->contact_map_.setContactPose(ee_name, pose_ref.translation());

  CostStack *cs = get_cost_stack(t);
  QuadraticResidualCost *qrc1 = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at("linear_acc_cost")]);
  QuadraticResidualCost *qrc2 = dynamic_cast<QuadraticResidualCost *>(
      &*cs->components_[cost_map_.at("angular_acc_cost")]);
  CentroidalAccelerationResidual *car =
      dynamic_cast<CentroidalAccelerationResidual *>(&*qrc1->residual_);
  AngularAccelerationResidual *aar =
      dynamic_cast<AngularAccelerationResidual *>(&*qrc2->residual_);
  car->contact_map_.setContactPose(ee_name, pose_ref.translation());
  aar->contact_map_.setContactPose(ee_name, pose_ref.translation());
}

pinocchio::SE3
CentroidalProblem::get_reference_pose(const std::size_t t,
                                      const std::string &ee_name) {
  if (t >= problem_->stages_.size()) {
    throw std::runtime_error("Stage index exceeds stage vector size");
  }
  IntegratorSemiImplEuler *dyn = dynamic_cast<IntegratorSemiImplEuler *>(
      &*problem_->stages_[t]->dynamics_);
  CentroidalFwdDynamics *cent_dyn =
      dynamic_cast<CentroidalFwdDynamics *>(&*dyn->ode_);

  pinocchio::SE3 pose = pinocchio::SE3::Identity();
  pose.translation() = cent_dyn->contact_map_.getContactPose(ee_name);

  return pose;
}

void CentroidalProblem::set_reference_forces(
    const std::size_t t,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  compute_control_from_forces(force_refs);
  set_reference_control(t, control_ref_);
}

void CentroidalProblem::set_reference_force(const std::size_t t,
                                            const std::string &ee_name,
                                            const Eigen::VectorXd &force_ref) {
  std::vector<std::string> hname = handler_.get_ee_names();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();
  control_ref_.segment(id * settings_.force_size, settings_.force_size) =
      force_ref;
  set_reference_control(t, control_ref_);
}

Eigen::VectorXd
CentroidalProblem::get_reference_force(const std::size_t t,
                                       const std::string &ee_name) {
  std::vector<std::string> hname = handler_.get_ee_names();
  std::vector<std::string>::iterator it =
      std::find(hname.begin(), hname.end(), ee_name);
  long id = it - hname.begin();

  return get_reference_control(t).segment(id * settings_.force_size,
                                          settings_.force_size);
}

Eigen::VectorXd
CentroidalProblem::get_x0_from_multibody(const Eigen::VectorXd &x_multibody) {
  if (x_multibody.size() != handler_.get_x0().size()) {
    throw std::runtime_error("x_multibody is of incorrect size");
  }
  handler_.updateInternalData(x_multibody);
  Eigen::VectorXd x0(9);
  x0.setZero();
  x0.head(3) = handler_.get_com_position();
  x0.segment(3, 3) = handler_.get_rdata().hg.linear();
  x0.tail(3) = handler_.get_rdata().hg.angular();

  return x0;
}

CostStack CentroidalProblem::create_terminal_cost() {
  auto ter_space = VectorSpace(nx_);
  auto term_cost = CostStack(ter_space, nu_);
  auto linear_mom = LinearMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
  auto angular_mom = AngularMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
  term_cost.addCost(
      QuadraticResidualCost(ter_space, linear_mom, settings_.w_linear_mom));
  term_cost.addCost(
      QuadraticResidualCost(ter_space, angular_mom, settings_.w_angular_mom));

  return term_cost;

} // namespace simple_mpc

} // namespace simple_mpc
