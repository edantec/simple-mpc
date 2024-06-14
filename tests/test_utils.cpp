
#include <boost/test/unit_test.hpp>

#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

using namespace simple_mpc;

RobotHandler getTalosHandler() {
  RobotHandlerSettings settings;
  settings.urdf_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
  settings.srdf_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";

  settings.controlled_joints_names = {
      "root_joint",        "leg_left_1_joint",  "leg_left_2_joint",
      "leg_left_3_joint",  "leg_left_4_joint",  "leg_left_5_joint",
      "leg_left_6_joint",  "leg_right_1_joint", "leg_right_2_joint",
      "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint",
      "leg_right_6_joint", "torso_1_joint",     "torso_2_joint",
      "arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",
      "arm_left_4_joint",  "arm_right_1_joint", "arm_right_2_joint",
      "arm_right_3_joint", "arm_right_4_joint",
  };
  settings.end_effector_names = {"left_sole_link", "right_sole_link"};
  settings.base_configuration = "half_sitting";
  settings.root_name = "root_joint";
  settings.loadRotor = true;

  RobotHandler handler(settings);

  return handler;
}

FullDynamicsSettings getFullDynamicsSettings(RobotHandler handler) {
  int nv = handler.get_rmodel().nv;
  int nu = nv - 6;

  FullDynamicsSettings settings;
  settings.x0 = handler.get_x0();
  settings.u0 = Eigen::VectorXd::Zero(nu);
  settings.DT = 0.01;
  settings.w_x = Eigen::MatrixXd::Identity(nv * 2, nv * 2);
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu);
  settings.w_cent = Eigen::MatrixXd::Identity(6, 6);
  settings.w_centder = Eigen::MatrixXd::Identity(6, 6);
  settings.gravity << 0, 0, 9;
  settings.force_size = 6;
  settings.w_forces = Eigen::MatrixXd::Identity(6, 6);
  settings.w_frame = Eigen::MatrixXd::Identity(6, 6);
  settings.umin =
      -handler.get_rmodel().effortLimit.tail(handler.get_rmodel().nv - 6);
  settings.umin =
      handler.get_rmodel().effortLimit.tail(handler.get_rmodel().nv - 6);
  settings.qmin = -Eigen::VectorXd::Ones(handler.get_rmodel().nv);
  settings.qmax = Eigen::VectorXd::Ones(handler.get_rmodel().nv);

  return settings;
}
