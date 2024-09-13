///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "simple-mpc/lowlevel-control.hpp"
#include <proxsuite/proxqp/settings.hpp>

namespace simple_mpc {

Lowlevel::Lowlevel() {}

Lowlevel::Lowlevel(const IDSettings &settings, const pinocchio::Model &model) {
  initialize(settings, model);
}

void Lowlevel::initialize(const IDSettings &settings,
                          const pinocchio::Model &model) {
  settings_ = settings;
  model_ = model;

  force_dim_ = (int)settings.force_size * settings.nk;

  int n = 2 * model_.nv - 6 + force_dim_;
  int neq = model_.nv + force_dim_;
  int nin = 9 * settings.nk;

  baum_gains_ << settings_.kd, settings_.kd, settings_.kd;
  A_.resize(neq, n);
  A_.setZero();
  b_.resize(neq);
  b_.setZero();
  l_.resize(nin);
  l_.setZero();
  C_.resize(nin, n);
  C_.setZero();
  S_.resize(model_.nv, model_.nv - 6);
  S_.setZero();
  S_.bottomRows(model_.nv - 6).diagonal().setOnes();
  Cmin_.resize(9, settings.force_size);
  if (settings.force_size == 3) {
    Cmin_ << -1, 0, settings.mu, 1, 0, settings.mu, -1, 0, settings.mu, 1, 0,
        settings.mu, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1;
  } else {
    Cmin_ << -1, 0, settings.mu, 0, 0, 0, 1, 0, settings.mu, 0, 0, 0, -1, 0,
        settings.mu, 0, 0, 0, 1, 0, settings.mu, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, settings.Wfoot, -1, 0, 0, 0, 0, settings.Wfoot, 1, 0, 0, 0, 0,
        settings.Lfoot, 0, -1, 0, 0, 0, settings.Lfoot, 0, 1, 0;
  }
  Jc_.resize(force_dim_, model_.nv);
  Jc_.setZero();
  gamma_.resize(force_dim_);
  gamma_.setZero();
  Jdot_.resize(6, model_.nv);
  Jdot_.setZero();

  u_ = Eigen::VectorXd::Ones(9 * settings.nk) * 100000;
  g_ = Eigen::VectorXd::Zero(n);
  H_ = Eigen::MatrixXd::Zero(n, n);
  H_.topLeftCorner(model_.nv, model_.nv).diagonal() =
      Eigen::VectorXd::Ones(model_.nv) * settings.w_acc;
  H_.block(model_.nv, model_.nv, force_dim_, force_dim_).diagonal() =
      Eigen::VectorXd::Ones(force_dim_) * settings.w_force;

  solved_forces_.resize(force_dim_);
  solved_acc_.resize(model_.nv);
  solved_torque_.resize(model_.nv - 6);

  qp_ = std::make_shared<proxqp::dense::QP<double>>(
      n, neq, nin, false, proxqp::HessianType::Dense,
      proxqp::DenseBackend::PrimalDualLDLT);
  qp_->settings.eps_abs = 1e-3;
  qp_->settings.eps_rel = 0.0;
  qp_->settings.primal_infeasibility_solving = true;
  qp_->settings.check_duality_gap = true;
  qp_->settings.verbose = settings.verbose;
  qp_->settings.max_iter = 10;
  qp_->settings.max_iter_in = 10;
  qp_->init(H_, g_, A_, b_, C_, l_, u_);
}

void Lowlevel::computeMatrice(pinocchio::Data &data,
                              std::vector<bool> contact_state,
                              Eigen::VectorXd v, Eigen::VectorXd a,
                              Eigen::VectorXd forces, Eigen::MatrixXd M) {

  Jc_.setZero();
  gamma_.setZero();
  l_.setZero();
  C_.setZero();
  for (long i = 0; i < settings_.nk; i++) {
    if (contact_state[(size_t)i]) {
      Jvel_ = getFrameVelocity(model_, data, settings_.contact_ids[(size_t)i],
                               pinocchio::LOCAL_WORLD_ALIGNED);
      getFrameJacobianTimeVariation(model_, data,
                                    settings_.contact_ids[(size_t)i],
                                    LOCAL_WORLD_ALIGNED, Jdot_);
      Jc_.middleRows(i * settings_.force_size, settings_.force_size) =
          getFrameJacobian(model_, data, settings_.contact_ids[(size_t)i],
                           LOCAL_WORLD_ALIGNED)
              .topRows(settings_.force_size);
      gamma_.segment(i * settings_.force_size, settings_.force_size) =
          Jdot_.topRows(settings_.force_size) * v;
      gamma_.segment(i * settings_.force_size, 3) +=
          baum_gains_.transpose() * Jvel_.linear() +
          baum_gains_.transpose() * Jvel_.angular();

      l_.segment(i * 9, 9) << forces[i * settings_.force_size] -
                                  forces[i * settings_.force_size + 2] *
                                      settings_.mu,
          -forces[i * settings_.force_size] -
              forces[i * settings_.force_size + 2] * settings_.mu,
          forces[i * settings_.force_size + 1] -
              forces[i * settings_.force_size + 2] * settings_.mu,
          -forces[i * settings_.force_size + 1] -
              forces[i * settings_.force_size + 2] * settings_.mu,
          -forces[i * settings_.force_size + 2],
          forces[i * settings_.force_size + 3] -
              forces[i * settings_.force_size + 2] * settings_.Wfoot,
          -forces[i * settings_.force_size + 3] -
              forces[i * settings_.force_size + 2] * settings_.Wfoot,
          forces[i * settings_.force_size + 4] -
              forces[i * settings_.force_size + 2] * settings_.Lfoot,
          -forces[i * settings_.force_size + 4] -
              forces[i * settings_.force_size + 2] * settings_.Lfoot;

      C_.block(i * 9, model_.nv + i * settings_.force_size, 9,
               settings_.force_size) = Cmin_;
    }
  }

  A_.topLeftCorner(model_.nv, model_.nv) = M;
  A_.block(0, model_.nv, model_.nv, force_dim_) = -Jc_.transpose();
  A_.topRightCorner(model_.nv, model_.nv - 6) = -S_;
  A_.bottomLeftCorner(force_dim_, model_.nv) = Jc_;

  b_.head(model_.nv) = -data.nle - M * a + Jc_.transpose() * forces;
  b_.tail(force_dim_) = -gamma_ - Jc_ * a;
}

void Lowlevel::solve_qp(pinocchio::Data &data, std::vector<bool> contact_state,
                        Eigen::VectorXd v, Eigen::VectorXd a,
                        Eigen::VectorXd forces, Eigen::MatrixXd M) {

  computeMatrice(data, contact_state, v, a, forces, M);
  qp_->update(H_, g_, A_, b_, C_, l_, u_, false);
  qp_->solve();

  solved_acc_ = a + qp_->results.x.head(model_.nv);
  solved_forces_ = forces + qp_->results.x.segment(model_.nv, force_dim_);
  solved_torque_ = qp_->results.x.tail(model_.nv - 6);
}

} // namespace simple_mpc
