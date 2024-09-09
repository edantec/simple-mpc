#include <boost/test/unit_test.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
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

RobotHandler getSoloHandler() {
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
  settings.w_x.diagonal() << 0, 0, 0, 100, 100, 100, // Base pos/ori
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,                  // Left leg
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,                  // Right leg
      10, 10,                                        // Torso
      1, 1, 1, 1,                                    // Left arm
      1, 1, 1, 1,                                    // Right arm
      1, 1, 1, 1, 1, 1,                              // Base pos/ori vel
      0.1, 0.1, 0.1, 0.1, 0.01, 0.01,                // Left leg vel
      0.1, 0.1, 0.1, 0.1, 0.01, 0.01,                // Right leg vel
      10, 10,                                        // Torso vel
      1, 1, 1, 1,                                    // Left arm vel
      1, 1, 1, 1;                                    // Right arm vel
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;

  settings.w_cent = Eigen::MatrixXd::Identity(6, 6);
  settings.w_cent.diagonal() << 0, 0, 0, 0.1, 0.1, 100;

  settings.gravity << 0, 0, -9.81;
  settings.force_size = 6;
  settings.w_forces = Eigen::MatrixXd::Identity(6, 6);
  settings.w_forces.diagonal() << 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001;
  settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 2000;
  settings.umin =
      -handler.get_rmodel().effortLimit.tail(handler.get_rmodel().nv - 6);
  settings.umax =
      handler.get_rmodel().effortLimit.tail(handler.get_rmodel().nv - 6);
  settings.qmin =
      handler.get_rmodel().lowerPositionLimit.tail(handler.get_rmodel().nv - 6);
  settings.qmax =
      handler.get_rmodel().upperPositionLimit.tail(handler.get_rmodel().nv - 6);

  return settings;
}

KinodynamicsSettings getKinodynamicsSettings(RobotHandler handler) {
  int nv = handler.get_rmodel().nv;
  int nu = nv + 6;

  KinodynamicsSettings settings;
  settings.x0 = handler.get_x0();
  settings.u0 = Eigen::VectorXd::Zero(nu);
  settings.DT = 0.01;
  settings.w_x = Eigen::MatrixXd::Identity(nv * 2, nv * 2);
  settings.w_x.diagonal() << 0, 0, 1000, 1000, 1000, 1000, // Base pos/ori
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,                        // Left leg
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,                        // Right leg
      100, 1000,                                           // Torso
      10, 10, 10, 10,                                      // Left arm
      10, 10, 10, 10,                                      // Right arm
      0.1, 0.1, 0.1, 1000, 1000, 1000,                     // Base pos/ori vel
      1, 1, 1, 1, 1, 1,                                    // Left leg vel
      1, 1, 1, 1, 1, 1,                                    // Right leg vel
      0.1, 100,                                            // Torso vel
      10, 10, 10, 10,                                      // Left arm vel
      10, 10, 10, 10;                                      // Right arm vel
  settings.w_x.diagonal() *= 10;
  Eigen::VectorXd w_linforce(3);
  Eigen::VectorXd w_angforce(3);
  Eigen::VectorXd w_ujoint = Eigen::VectorXd::Ones(nv - 6) * 1e-3;
  w_linforce << 0.001, 0.001, 0.001;
  w_angforce << 1, 1, 1;
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu);
  settings.w_u.diagonal() << w_linforce, w_angforce, w_linforce, w_angforce,
      w_ujoint;
  settings.w_cent = Eigen::MatrixXd::Identity(6, 6);
  settings.w_cent.diagonal() << 0, 0, 0, 0.1, 0.1, 0.1;
  settings.w_centder = Eigen::MatrixXd::Identity(6, 6);
  settings.w_centder.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  settings.gravity << 0, 0, -9.81;
  settings.force_size = 6;
  settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 50000;
  settings.qmin =
      handler.get_rmodel().lowerPositionLimit.tail(handler.get_rmodel().nv - 6);
  settings.qmax =
      handler.get_rmodel().upperPositionLimit.tail(handler.get_rmodel().nv - 6);

  return settings;
}

CentroidalSettings getCentroidalSettings(RobotHandler handler) {
  int nx = 9;
  int nu = 6 * 2;

  CentroidalSettings settings;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  x0.head(3) = handler.get_com_position();
  settings.x0 = x0;
  settings.u0 = Eigen::VectorXd::Zero(nu);
  settings.DT = 0.01;
  settings.w_x_ter = Eigen::MatrixXd::Identity(nx, nx);
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu);

  settings.w_linear_mom = Eigen::MatrixXd::Identity(3, 3);
  settings.w_linear_mom.diagonal() << 0.01, 0.01, 100;
  settings.w_angular_mom = Eigen::MatrixXd::Identity(3, 3);
  settings.w_angular_mom.diagonal() << 0.1, 0.1, 1000;
  settings.w_linear_acc = Eigen::MatrixXd::Identity(3, 3);
  settings.w_linear_acc.diagonal() << 0.01, 0.01, 0.01;
  settings.w_angular_acc = Eigen::MatrixXd::Identity(3, 3);
  settings.w_angular_acc.diagonal() << 0.01, 0.01, 0.01;
  settings.gravity << 0, 0, -9.81;
  settings.force_size = 6;

  return settings;
}
