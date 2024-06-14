
#include <boost/test/unit_test.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(robot_handler)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(build_talos) {
  RobotHandler handler = getTalosHandler();

  BOOST_CHECK_EQUAL(handler.get_rmodel().nq, 29);
  BOOST_CHECK_EQUAL(handler.get_rmodel().nv, 28);
  BOOST_CHECK_EQUAL(handler.get_mass(), 90.272192000000018);

  Eigen::VectorXd q1(29);
  q1 << 0, 0, 0, 0, 0, 0, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.1,
      0.1, 0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

  handler.set_q0(q1);
  BOOST_CHECK_EQUAL(handler.get_q0(), q1);
  BOOST_CHECK_EQUAL(handler.get_ee_name(1), "right_sole_link");

  Eigen::Vector3d com = handler.get_com_position();
}

BOOST_AUTO_TEST_CASE(build_solo) {
  RobotHandlerSettings settings;
  settings.urdf_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
  settings.srdf_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf";

  settings.controlled_joints_names = {
      "root_joint", "FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE",
      "HL_HAA",     "HL_HFE", "HL_KFE", "HR_HAA", "HR_HFE", "HR_KFE",
  };
  settings.end_effector_names = {"FR_FOOT", "FL_FOOT", "HL_FOOT", "HR_FOOT"};
  settings.base_configuration = "straight_standing";
  settings.root_name = "root_joint";

  RobotHandler handler(settings);

  BOOST_CHECK_EQUAL(handler.get_rmodel().nq, 19);
  BOOST_CHECK_EQUAL(handler.get_rmodel().nv, 18);
  BOOST_CHECK_EQUAL(handler.get_mass(), 2.5000027900000004);

  Eigen::VectorXd q1(19);
  q1.setZero();
  q1[6] = 1;

  handler.set_q0(q1);
  BOOST_CHECK_EQUAL(handler.get_q0(), q1);
  BOOST_CHECK_EQUAL(handler.get_ee_name(1), "FL_FOOT");

  Eigen::Vector3d com = handler.get_com_position();
}

BOOST_AUTO_TEST_SUITE_END()
