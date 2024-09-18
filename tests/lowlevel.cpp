
#include <boost/test/unit_test.hpp>
#include <proxsuite-nlp/manifold-base.hpp>

#include "simple-mpc/lowlevel-control.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(lowlevel)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(ID_solver) {
  RobotHandler handler = getTalosHandler();

  IDSettings settings;
  settings.contact_ids = handler.getFeetIds();
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;
  settings.kd = 10;
  settings.w_force = 1000;
  settings.w_acc = 1;
  settings.verbose = false;

  IDSolver ID_solver(settings, handler.getModel());

  std::vector<bool> contact_states;
  contact_states.push_back(true);
  contact_states.push_back(true);

  Eigen::VectorXd v = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(6 * 2);

  Eigen::MatrixXd M = handler.getMassMatrix();
  pinocchio::Data rdata = handler.getData();
  ID_solver.solve_qp(rdata, contact_states, v, a, forces, M);
}

BOOST_AUTO_TEST_CASE(IKID_solver) {
  RobotHandler handler = getTalosHandler();
  std::vector<FrameIndex> vec_base;
  std::vector<Eigen::VectorXd> Kp;
  std::vector<Eigen::VectorXd> Kd;

  Eigen::VectorXd g_q(handler.getModel().nv);
  g_q << 0, 0, 0, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10,
      100, 100, 100, 100, 100, 100, 100, 100;

  double g_p = 400;
  double g_b = 10;

  Kp.push_back(g_q);
  Kp.push_back(Eigen::VectorXd::Constant(6, g_p));
  Kp.push_back(Eigen::VectorXd::Constant(3, g_b));

  Kd.push_back(2 * g_q.array().sqrt());
  Kd.push_back(Eigen::VectorXd::Constant(6, 2 * sqrt(g_p)));
  Kd.push_back(Eigen::VectorXd::Constant(3, 2 * sqrt(g_b)));

  vec_base.push_back(handler.getRootId());
  IKIDSettings settings;
  settings.contact_ids = handler.getFeetIds();
  settings.fixed_frame_ids = vec_base;
  settings.x0 = handler.getState();
  settings.Kp_gains = Kp, settings.Kd_gains = Kd, settings.dt = 0.01,
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;
  settings.w_qref = 500;
  settings.w_footpose = 50000;
  settings.w_centroidal = 10;
  settings.w_centroidal = 10;
  settings.w_baserot = 1000;
  settings.w_force = 100;
  settings.verbose = false;

  IKIDSolver IKID_solver(settings, handler.getModel());

  std::vector<bool> contact_states;
  contact_states.push_back(true);
  contact_states.push_back(true);

  std::vector<pinocchio::SE3> foot_refs;
  std::vector<pinocchio::SE3> foot_refs_next;
  for (auto const &name : handler.getFeetNames()) {
    pinocchio::SE3 foot_ref = handler.getFootPose(name);
    foot_refs.push_back(foot_ref);
    foot_ref.translation()[0] += 0.1;

    foot_refs_next.push_back(foot_ref);
  }

  Eigen::VectorXd dq = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd dv = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd dH = Eigen::VectorXd::Random(6);
  Eigen::VectorXd x0 = handler.getState();
  Eigen::VectorXd xm = handler.getState();
  Eigen::VectorXd new_q = pinocchio::integrate(
      handler.getModel(), x0.head(handler.getModel().nq), dq);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(6 * 2);
  xm << new_q, dv;

  Eigen::MatrixXd M = handler.getMassMatrix();
  pinocchio::Data rdata = handler.getData();
  IKID_solver.solve_qp(rdata, contact_states, xm, forces, foot_refs,
                       foot_refs_next, dH, M);
}

BOOST_AUTO_TEST_SUITE_END()
