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

IDSolver::IDSolver() {}

IDSolver::IDSolver(const IDSettings &settings, const pinocchio::Model &model) {
  initialize(settings, model);
}

void IDSolver::initialize(const IDSettings &settings,
                          const pinocchio::Model &model) {
  settings_ = settings;
  model_ = model;

  nk_ = (int)settings.contact_ids.size();
  force_dim_ = (int)settings.force_size * nk_;

  int n = 2 * model_.nv - 6 + force_dim_;
  int neq = model_.nv + force_dim_;
  int nin = 9 * nk_;

  baum_gains_.setZero();
  baum_gains_.diagonal() << settings_.kd, settings_.kd, settings_.kd;
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

  for (long i = 0; i < nk_; i++) {
    C_.block(i * 9, model_.nv + i * settings_.force_size, 9,
             settings_.force_size) = Cmin_;
  }
  Jc_.resize(force_dim_, model_.nv);
  Jc_.setZero();
  gamma_.resize(force_dim_);
  gamma_.setZero();
  Jdot_.resize(6, model_.nv);
  Jdot_.setZero();

  u_ = Eigen::VectorXd::Ones(9 * nk_) * 100000;
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

void IDSolver::computeMatrice(pinocchio::Data &data,
                              const std::vector<bool> &contact_state,
                              const Eigen::VectorXd &v,
                              const Eigen::VectorXd &a,
                              const Eigen::VectorXd &forces,
                              const Eigen::MatrixXd &M) {

  Jc_.setZero();
  gamma_.setZero();
  l_.setZero();
  C_.setZero();
  for (long i = 0; i < nk_; i++) {
    Jdot_.setZero();
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
      /* gamma_.segment(i * settings_.force_size, 3) +=
          baum_gains_ * Jvel_.linear() + baum_gains_ * Jvel_.angular(); */

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

void IDSolver::solve_qp(pinocchio::Data &data,
                        const std::vector<bool> &contact_state,
                        const Eigen::VectorXd &v, const Eigen::VectorXd &a,
                        const Eigen::VectorXd &forces,
                        const Eigen::MatrixXd &M) {

  computeMatrice(data, contact_state, v, a, forces, M);
  qp_->update(H_, g_, A_, b_, C_, l_, u_, false);
  qp_->solve();

  solved_acc_ = a + qp_->results.x.head(model_.nv);
  solved_forces_ = forces + qp_->results.x.segment(model_.nv, force_dim_);
  solved_torque_ = qp_->results.x.tail(model_.nv - 6);
}

IKIDSolver::IKIDSolver() {}

IKIDSolver::IKIDSolver(const IKIDSettings &settings,
                       const pinocchio::Model &model) {
  initialize(settings, model);
}

void IKIDSolver::initialize(const IKIDSettings &settings,
                            const pinocchio::Model &model) {
  settings_ = settings;
  model_ = model;
  foot_diff_.resize(6);
  foot_diff_.setZero();
  dfoot_diff_.resize(6);
  dfoot_diff_.setZero();
  frame_diff_.resize(3);
  frame_diff_.setZero();
  dframe_diff_.resize(6);
  dframe_diff_.setZero();
  q_diff_.resize(model_.nv);
  q_diff_.setZero();
  dq_diff_.resize(model_.nv);
  dq_diff_.setZero();

  fs_ = (int)settings.force_size;
  nk_ = (int)settings.contact_ids.size();
  force_dim_ = fs_ * nk_;

  int n = 2 * model_.nv - 6 + force_dim_;
  int neq = model_.nv + force_dim_;
  int nin = 9 * nk_;

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
  l_box_.resize(n);
  l_box_.setOnes();
  l_box_ *= -10000;
  l_box_.tail(model.nv - 6) = -model.effortLimit.tail(model.nv - 6);
  u_box_.resize(n);
  u_box_.setOnes();
  u_box_ *= 10000;
  u_box_.tail(model.nv - 6) = model.effortLimit.tail(model.nv - 6);

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

  for (long i = 0; i < nk_; i++) {
    C_.block(i * 9, model_.nv + i * settings_.force_size, 9,
             settings_.force_size) = Cmin_;
  }
  Jfoot_.resize(6, model_.nv);
  Jfoot_.setZero();
  dJfoot_.resize(6, model_.nv);
  dJfoot_.setZero();
  Jframe_.resize(3, model_.nv);
  Jframe_.setZero();
  dJframe_.resize(6, model_.nv);
  dJframe_.setZero();

  u_ = Eigen::VectorXd::Ones(9 * nk_) * 100000;
  g_ = Eigen::VectorXd::Zero(n);
  H_ = Eigen::MatrixXd::Zero(n, n);
  H_.block(model_.nv, model_.nv, force_dim_, force_dim_).diagonal() =
      Eigen::VectorXd::Ones(force_dim_) * settings.w_force;

  solved_forces_.resize(force_dim_);
  solved_acc_.resize(model_.nv);
  solved_torque_.resize(model_.nv - 6);

  qp_ = std::make_shared<proxqp::dense::QP<double>>(
      n, neq, nin, true, proxqp::HessianType::Dense,
      proxqp::DenseBackend::PrimalDualLDLT);
  qp_->settings.eps_abs = 1e-3;
  qp_->settings.eps_rel = 0.0;
  qp_->settings.primal_infeasibility_solving = true;
  qp_->settings.check_duality_gap = true;
  qp_->settings.verbose = settings.verbose;
  qp_->settings.max_iter = 10;
  qp_->settings.max_iter_in = 10;

  qp_->init(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
}

void IKIDSolver::computeMatrice(
    pinocchio::Data &data, const std::vector<bool> &contact_state,
    const Eigen::VectorXd &x_measured, const Eigen::VectorXd &forces,
    const std::vector<pinocchio::SE3> foot_refs,
    const std::vector<pinocchio::SE3> foot_refs_next, const Eigen::VectorXd &dH,
    const Eigen::MatrixXd &M) {

  difference(model_, x_measured.head(model_.nq), settings_.x0.head(model_.nq),
             q_diff_);
  dq_diff_ = settings_.x0.tail(model_.nv), x_measured.tail(model_.nv);

  H_.topLeftCorner(model_.nv, model_.nv) =
      settings_.w_qref * Eigen::MatrixXd::Identity(model_.nv, model_.nv);
  H_.topLeftCorner(model_.nv, model_.nv) +=
      settings_.w_centroidal * data.Ag.transpose() * data.Ag;

  g_.head(model_.nv) =
      settings_.w_qref * (-settings_.Kp_gains[0].cwiseProduct(q_diff_) -
                          settings_.Kd_gains[0].cwiseProduct(dq_diff_));
  g_.head(model_.nv) -=
      settings_.w_centroidal *
      (dH - data.dAg * x_measured.tail(model_.nv)).transpose() * data.Ag;

  A_.topLeftCorner(model_.nv, model_.nv) = M;
  A_.topRightCorner(model_.nv, model_.nv - 6) = -S_;

  b_.head(model_.nv) = -data.nle;
  b_.tail(force_dim_).setZero();
  l_.setZero();
  C_.setZero();

  for (size_t i = 0; i < settings_.contact_ids.size(); i++) {
    FrameIndex id = settings_.contact_ids[i];
    Jfoot_ = getFrameJacobian(model_, data, id, LOCAL);
    getFrameJacobianTimeVariation(model_, data, settings_.contact_ids[0], LOCAL,
                                  dJfoot_);

    foot_diff_.head(3) =
        foot_refs[i].translation() - data.oMf[id].translation();
    foot_diff_.tail(3) =
        -log3(foot_refs[i].rotation().transpose() * data.oMf[id].rotation());

    dfoot_diff_.head(3) =
        (foot_refs_next[i].translation() - foot_refs[i].translation()) /
            settings_.dt -
        getFrameVelocity(model_, data, id, LOCAL).linear();
    dfoot_diff_.tail(3) = log3(foot_refs[i].rotation().transpose() *
                               foot_refs_next[i].rotation()) /
                              settings_.dt -
                          getFrameVelocity(model_, data, id, LOCAL).angular();

    H_.topLeftCorner(model_.nv, model_.nv) +=
        settings_.w_footpose * Jfoot_.transpose() * Jfoot_;

    g_.head(model_.nv) += settings_.w_footpose *
                          (Jfoot_ * x_measured.tail(model_.nv) -
                           settings_.Kp_gains[1].cwiseProduct(foot_diff_) -
                           settings_.Kd_gains[1].cwiseProduct(dfoot_diff_))
                              .transpose() *
                          Jfoot_;

    long il = (long)i;
    if (contact_state[i]) {
      A_.block(0, model_.nv + il * 6, model_.nv, 6) = -Jfoot_.transpose();
      A_.block(model_.nv + il * 6, 0, 6, model_.nv) = Jfoot_;
      b_.head(model_.nv) += Jfoot_.transpose() * forces.segment(il * 6, 6);
      b_.segment(model_.nv + il * 6, 6) = -dJfoot_ * x_measured.tail(model_.nv);

      l_.segment(il * 9, 9)
          << forces[il * fs_] - forces[il * fs_ + 2] * settings_.mu,
          -forces[il * fs_] - forces[il * fs_ + 2] * settings_.mu,
          forces[il * fs_ + 1] - forces[il * fs_ + 2] * settings_.mu,
          -forces[il * fs_ + 1] - forces[il * fs_ + 2] * settings_.mu,
          -forces[il * fs_ + 2],
          forces[il * fs_ + 3] - forces[il * fs_ + 2] * settings_.Wfoot,
          -forces[il * fs_ + 3] - forces[il * fs_ + 2] * settings_.Wfoot,
          forces[il * fs_ + 4] - forces[il * fs_ + 2] * settings_.Lfoot,
          -forces[il * fs_ + 4] - forces[il * fs_ + 2] * settings_.Lfoot;

      C_.block(il * 9, model_.nv + il * fs_, 9, fs_) = Cmin_;
    } else {
      A_.block(0, model_.nv, model_.nv + il * 6, 6).setZero();
      A_.block(model_.nv + il * 6, 0, 6, model_.nv).setZero();
    }
  }
  for (size_t i = 0; i < settings_.fixed_frame_ids.size(); i++) {
    FrameIndex id = settings_.fixed_frame_ids[i];
    Jframe_ = getFrameJacobian(model_, data, id, LOCAL).bottomRows(3);
    getFrameJacobianTimeVariation(model_, data, id, LOCAL, dJframe_);
    frame_diff_ = -log3(data.oMf[id].rotation());
    dframe_diff_ = -getFrameVelocity(model_, data, id, LOCAL).angular();

    H_.topLeftCorner(model_.nv, model_.nv) +=
        settings_.w_baserot * Jframe_.transpose() * Jframe_;
    g_.head(model_.nv) += settings_.w_baserot *
                          (dJframe_.bottomRows(3) * x_measured.tail(model_.nv) -
                           settings_.Kp_gains[2].cwiseProduct(frame_diff_) -
                           settings_.Kd_gains[2].cwiseProduct(dframe_diff_))
                              .transpose() *
                          Jframe_;
  }
}

void IKIDSolver::solve_qp(pinocchio::Data &data,
                          const std::vector<bool> &contact_state,
                          const Eigen::VectorXd &x_measured,
                          const Eigen::VectorXd &forces,
                          const std::vector<pinocchio::SE3> foot_refs,
                          const std::vector<pinocchio::SE3> foot_refs_next,
                          const Eigen::VectorXd &dH, const Eigen::MatrixXd &M) {

  computeMatrice(data, contact_state, x_measured, forces, foot_refs,
                 foot_refs_next, dH, M);
  qp_->update(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_, false);
  qp_->solve();

  solved_acc_ = qp_->results.x.head(model_.nv);
  solved_forces_ = forces + qp_->results.x.segment(model_.nv, force_dim_);
  solved_torque_ = qp_->results.x.tail(model_.nv - 6);
}

} // namespace simple_mpc
