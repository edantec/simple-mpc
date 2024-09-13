
#include <boost/test/unit_test.hpp>

#include "simple-mpc/lowlevel-control.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(robot_handler)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(ID_solver) {
  RobotHandler handler = getTalosHandler();

  IDSettings settings;
  settings.nk = 2;
  settings.contact_ids = handler.getFeetIds();
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;
  settings.kd = 10;
  settings.w_force = 1000;
  settings.w_acc = 1;
  settings.verbose = false;

  Lowlevel lowlevel_qp(settings, handler.getModel());

  std::vector<bool> contact_states;
  contact_states.push_back(true);
  contact_states.push_back(true);

  Eigen::VectorXd v = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(handler.getModel().nv);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(6 * 2);
}

BOOST_AUTO_TEST_SUITE_END()
