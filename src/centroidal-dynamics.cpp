///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/centroidal-dynamics.hpp"

namespace simple_mpc {
using namespace aligator;

CentroidalProblem::CentroidalProblem(const CentroidalSettings &settings,
                                     const RobotHandler &handler)
    : Base(handler), settings_(settings) {

  nx_ = 18;
  nu_ = (int)handler_.get_ee_names().size() * settings_.force_size;
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

void CentroidalProblem::create_problem(
    const Eigen::VectorXd &x0, const std::vector<ContactMap> &contact_sequence,
    const std::vector<std::map<std::string, Eigen::VectorXd>> &force_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models =
      create_stages(contact_sequence, force_sequence);
  problem_ = std::make_shared<TrajOptProblem>(x0, stage_models,
                                              create_terminal_cost());
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

void CentroidalProblem::set_reference_forces(
    const std::size_t t,
    const std::map<std::string, Eigen::VectorXd> &force_refs) {
  compute_control_from_forces(force_refs);
  set_reference_control(t, control_ref_);
}

void CentroidalProblem::set_reference_forces(const std::size_t t,
                                             const std::string &ee_name,
                                             Eigen::VectorXd &force_ref) {
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

CostStack CentroidalProblem::create_terminal_cost() {
  auto ter_space = VectorSpace(nx_);
  auto term_cost = CostStack(ter_space, nu_);
  term_cost.addCost(
      QuadraticStateCost(ter_space, nu_, settings_.x0, settings_.w_x));

  return term_cost;
}

} // namespace simple_mpc
