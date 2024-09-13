
#include <boost/test/unit_test.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(robot_handler)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(build_talos) {
  RobotHandler handler = getTalosHandler();

  /* BOOST_CHECK_EQUAL(handler.getModel().nq, 29);
  BOOST_CHECK_EQUAL(handler.getModel().nv, 28);
  BOOST_CHECK_EQUAL(handler.getMass(), 90.272192000000018);

  Eigen::VectorXd q1(29);
  q1 << 0, 0, 0, 0, 0, 0, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.1,
      0.1, 0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

  handler.updateConfiguration(q1, false);
  BOOST_CHECK_EQUAL(handler.getConfiguration(), q1);
  BOOST_CHECK_EQUAL(handler.getFootName(1), "right_sole_link");

  Eigen::Vector3d com = handler.getComPosition();
  pinocchio::SE3 pose = handler.getFootPose("right_sole_link"); */

  /* Eigen::VectorXd q2(29);
  q2 << 0, 0, 0, 0, 0, 0, 1, -0.1, 0.1, 0.1, -0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.1,
      -0.1, 0, -0.1, 0, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1;
  Eigen::VectorXd v = Eigen::VectorXd::Random(29);
  Eigen::VectorXd x(57);
  x << q2, v;
  handler.updateState(x, true);
  BOOST_CHECK_EQUAL(handler.getConfiguration(), q2); */

  // Eigen::MatrixXd M = handler.getMassMatrix();
}
/*
BOOST_AUTO_TEST_CASE(build_solo) {
  RobotHandler handler = getSoloHandler();

  BOOST_CHECK_EQUAL(handler.getModel().nq, 19);
  BOOST_CHECK_EQUAL(handler.getModel().nv, 18);
  BOOST_CHECK_EQUAL(handler.getMass(), 2.5000027900000004);

  Eigen::VectorXd q1(19);
  q1.setZero();
  q1[6] = 1;

  handler.updateConfiguration(q1, false);
  BOOST_CHECK_EQUAL(handler.getConfiguration(), q1);
  BOOST_CHECK_EQUAL(handler.getFootName(1), "FL_FOOT");

  Eigen::Vector3d com = handler.getComPosition();
  pinocchio::SE3 pose = handler.getFootPose("FL_FOOT");
} */

BOOST_AUTO_TEST_SUITE_END()
