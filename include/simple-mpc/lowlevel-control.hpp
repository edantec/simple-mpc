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
using Manifold = aligator::ManifoldAbstractTpl<double>;
/**
 * @brief Build a low-level control for kinodynamics
 * and centroidal MPC schemes
 */

struct IDSettings {
public:
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

struct IKIDSettings {
public:
  std::vector<Eigen::VectorXd> Kp_gains;          // Proportional gains
  std::vector<Eigen::VectorXd> Kd_gains;          // Derivative gains
  std::vector<pinocchio::FrameIndex> contact_ids; // Index of contacts
  std::vector<pinocchio::FrameIndex>
      fixed_frame_ids; // Index of frames kept fixed
  Eigen::VectorXd x0;  // Reference state
  double dt;           // Integration timestep
  double mu;           // Friction parameter
  double Lfoot;        // Half-length of foot (if contact 6D)
  double Wfoot;        // Half-width of foot (if contact 6D)
  long force_size;     // Dimension of contact forces
  double w_qref;       // Weight for configuration regularization
  double w_footpose;   // Weight for foot placement
  double w_centroidal; // Weight for CoM tracking
  double w_baserot;    // Weight for base rotation
  double w_force;      // Weight for force regularization
  bool verbose;        // Print solver information
};

class IDSolver {

protected:
  IDSettings settings_;
  std::shared_ptr<proxqp::dense::QP<double>> qp_;
  pinocchio::Model model_;
  int force_dim_;
  int nk_;

  Eigen::MatrixXd H_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd S_;
  Eigen::MatrixXd Cmin_;
  Eigen::VectorXd b_;
  Eigen::VectorXd g_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::Matrix3d baum_gains_;
  Motion Jvel_;

  Eigen::MatrixXd Jc_;
  Eigen::VectorXd gamma_;
  Eigen::MatrixXd Jdot_;

  // Internal matrix computation
  void computeMatrice(pinocchio::Data &data,
                      const std::vector<bool> &contact_state,
                      const Eigen::VectorXd &v, const Eigen::VectorXd &a,
                      const Eigen::VectorXd &forces, const Eigen::MatrixXd &M);

public:
  IDSolver();
  IDSolver(const IDSettings &settings, const pinocchio::Model &model);
  void initialize(const IDSettings &settings, const pinocchio::Model &model);

  void solve_qp(pinocchio::Data &data, const std::vector<bool> &contact_state,
                const Eigen::VectorXd &v, const Eigen::VectorXd &a,
                const Eigen::VectorXd &forces, const Eigen::MatrixXd &M);
  proxqp::dense::Model<double> getQP() { return qp_->model; }
  Eigen::MatrixXd getA() { return qp_->model.A; }
  Eigen::MatrixXd getH() { return qp_->model.H; }
  Eigen::MatrixXd getC() { return qp_->model.C; }
  Eigen::VectorXd getg() { return qp_->model.g; }
  Eigen::VectorXd getb() { return qp_->model.b; }

  // QP results
  Eigen::VectorXd solved_forces_;
  Eigen::VectorXd solved_acc_;
  Eigen::VectorXd solved_torque_;
};

class IKIDSolver {

protected:
  IKIDSettings settings_;
  std::shared_ptr<proxqp::dense::QP<double>> qp_;
  pinocchio::Model model_;
  int force_dim_;
  int nk_;
  int fs_;

  Eigen::MatrixXd H_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd S_;
  Eigen::MatrixXd Cmin_;
  Eigen::VectorXd b_;
  Eigen::VectorXd g_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::VectorXd l_box_;
  Eigen::VectorXd u_box_;
  Motion Jvel_;

  Eigen::MatrixXd Jfoot_;
  Eigen::MatrixXd dJfoot_;
  Eigen::MatrixXd Jframe_;
  Eigen::MatrixXd dJframe_;

  Eigen::VectorXd foot_diff_;
  Eigen::VectorXd dfoot_diff_;
  Eigen::VectorXd frame_diff_;
  Eigen::VectorXd dframe_diff_;
  Eigen::VectorXd q_diff_;
  Eigen::VectorXd dq_diff_;

  // Internal matrix computation
  void computeMatrice(pinocchio::Data &data,
                      const std::vector<bool> &contact_state,
                      const Eigen::VectorXd &x_measured,
                      const Eigen::VectorXd &forces,
                      const std::vector<pinocchio::SE3> foot_refs,
                      const std::vector<pinocchio::SE3> foot_refs_next,
                      const Eigen::VectorXd &dH, const Eigen::MatrixXd &M);

public:
  IKIDSolver();
  IKIDSolver(const IKIDSettings &settings, const pinocchio::Model &model);
  void initialize(const IKIDSettings &settings, const pinocchio::Model &model);

  void solve_qp(pinocchio::Data &data, const std::vector<bool> &contact_state,
                const Eigen::VectorXd &x_measured,
                const Eigen::VectorXd &forces,
                const std::vector<pinocchio::SE3> foot_refs,
                const std::vector<pinocchio::SE3> foot_refs_next,
                const Eigen::VectorXd &dH, const Eigen::MatrixXd &M);
  proxqp::dense::Model<double> getQP() { return qp_->model; }
  // QP results
  Eigen::VectorXd solved_forces_;
  Eigen::VectorXd solved_acc_;
  Eigen::VectorXd solved_torque_;
};

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_
