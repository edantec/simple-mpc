///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_
#define SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_

#include "simple-mpc/fwd.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <pinocchio/multibody/fwd.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace simple_mpc {
using namespace proxsuite;
using namespace pinocchio;
/**
 * @brief Build a low-level control for kinodynamics
 * and centroidal MPC schemes
 */

struct IDSettings {
public:
  int nk;                                         // Number of contacts
  std::vector<pinocchio::FrameIndex> contact_ids; // Index of contacts
  double mu;                                      // Friction parameter
  double Lfoot;    // Half-length of foot (if contact 6D)
  double Wfoot;    // Half-width of foot (if contact 6D)
  long force_size; // Dimension of contact forces
  double kd;       // Baumgarte coefficient
  double w_force;  // Weight for force regularization
  double w_acc;    // Weight for acceleration regularization
  bool verbose;    // Print solver information
};
class Lowlevel {

protected:
  IDSettings settings_;
  std::shared_ptr<proxqp::dense::QP<double>> qp_;
  pinocchio::Model model_;
  int force_dim_;

  Eigen::MatrixXd H_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd S_;
  Eigen::MatrixXd Cmin_;
  Eigen::VectorXd b_;
  Eigen::VectorXd g_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::Vector3d baum_gains_;
  Motion Jvel_;

  Eigen::MatrixXd Jc_;
  Eigen::VectorXd gamma_;
  Eigen::MatrixXd Jdot_;

  Eigen::VectorXd solved_forces_;
  Eigen::VectorXd solved_acc_;
  Eigen::VectorXd solved_torque_;

  // Internal matrix computation
  void computeMatrice(pinocchio::Data &data, std::vector<bool> contact_state,
                      Eigen::VectorXd v, Eigen::VectorXd a,
                      Eigen::VectorXd forces, Eigen::MatrixXd M);

public:
  Lowlevel();
  Lowlevel(const IDSettings &settings, const pinocchio::Model &model);
  void initialize(const IDSettings &settings, const pinocchio::Model &model);

  void solve_qp(pinocchio::Data &data, std::vector<bool> contact_state,
                Eigen::VectorXd v, Eigen::VectorXd a, Eigen::VectorXd forces,
                Eigen::MatrixXd M);
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_
