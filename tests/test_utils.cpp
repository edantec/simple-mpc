#include <boost/test/unit_test.hpp>

#include "simple-mpc/robot-handler.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/centroidal-dynamics.hpp"
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

using namespace simple_mpc;

RobotModelHandler getTalosModelHandler() {
    // Load pinocchio model from example robot data
    Model model;
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
    std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    srdf::loadReferenceConfigurations(model, srdf_path, false);
    srdf::loadRotorParameters(model, srdf_path, false);

    // Lock joint list
    const std::vector<std::string> controlled_joints_names { "universe",
        "root_joint",        "leg_left_1_joint",  "leg_left_2_joint",
        "leg_left_3_joint",  "leg_left_4_joint",  "leg_left_5_joint",
        "leg_left_6_joint",  "leg_right_1_joint", "leg_right_2_joint",
        "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint",
        "leg_right_6_joint", "torso_1_joint",     "torso_2_joint",
        "arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",
        "arm_left_4_joint",  "arm_right_1_joint", "arm_right_2_joint",
        "arm_right_3_joint", "arm_right_4_joint",
    };

    std::vector<std::string> locked_joints_names {model.names};
    locked_joints_names.erase(
        std::remove_if(locked_joints_names.begin(), locked_joints_names.end(),
            [&controlled_joints_names](const std::string& name)
            {
                return std::find(controlled_joints_names.begin(), controlled_joints_names.end(), name) != controlled_joints_names.end();
            }
        ), locked_joints_names.end()
    );


    // Actually create handler
    std::string base_joint = "root_joint";
    RobotModelHandler handler(model, "half_sitting", base_joint, locked_joints_names);

    // Add feet
    handler.addFoot("left_sole_link", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(0., 0.1, 0.)));
    handler.addFoot("right_sole_link", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(0., -0.1, 0.)));

    return handler;
}

RobotModelHandler getSoloHandler() {
    // Load pinocchio model from example robot data
    Model model;
    const std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
    const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    srdf::loadReferenceConfigurations(model, srdf_path, false);

    // Actually create handler
    std::string base_joint = "root_joint";
    RobotModelHandler handler(model, "straight_standing", base_joint);

    // Add feet
    handler.addFoot("FR_FOOT", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(0.1, -0.1, 0.)));
    handler.addFoot("FL_FOOT", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(0.1, 0.1, 0.)));
    handler.addFoot("HR_FOOT", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(-0.1, -0.1, 0.)));
    handler.addFoot("HL_FOOT", base_joint, SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(-0.1, 0.1, 0.)));

    return handler;
}

FullDynamicsSettings getFullDynamicsSettings(RobotModelHandler model_handler) {
  int nv = model_handler.getModel().nv;
  int nu = nv - 6;

  FullDynamicsSettings settings;
  settings.timestep = 0.01;
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
  settings.w_cent.diagonal() << 0, 0, 10, 0, 0, 10;

  settings.gravity << 0, 0, -9.81;
  settings.force_size = 6;
  settings.w_forces = Eigen::MatrixXd::Identity(6, 6);
  settings.w_forces.diagonal() << 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
      0.0001;
  settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 2000;
  settings.umin =
      model_handler.getModel().lowerEffortLimit.tail(model_handler.getModel().nv - 6);
  settings.umax =
      model_handler.getModel().upperEffortLimit.tail(model_handler.getModel().nv - 6);
  settings.qmin =
      model_handler.getModel().lowerPositionLimit.tail(model_handler.getModel().nv - 6);
  settings.qmax =
      model_handler.getModel().upperPositionLimit.tail(model_handler.getModel().nv - 6);
      handler.getModel().upperPositionLimit.tail(handler.getModel().nv - 6);
  settings.Kp_correction = Eigen::VectorXd::Ones(6);
  settings.Kd_correction = Eigen::VectorXd::Ones(6);
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.torque_limits = true;
  settings.kinematics_limits = true;
  settings.force_cone = true;

  return settings;
}

KinodynamicsSettings getKinodynamicsSettings(RobotModelHandler model_handler) {
  int nv = model_handler.getModel().nv;
  int nu = nv + 6;

  KinodynamicsSettings settings;
  settings.timestep = 0.01;
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
      model_handler.getModel().lowerPositionLimit.tail(model_handler.getModel().nv - 6);
  settings.qmax =
      model_handler.getModel().upperPositionLimit.tail(model_handler.getModel().nv - 6);
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.kinematics_limits = true;
  settings.force_cone = true;

  return settings;
}

CentroidalSettings getCentroidalSettings() {
  int nu = 6 * 2;

  CentroidalSettings settings;
  settings.timestep = 0.01;
  settings.w_com = Eigen::MatrixXd::Identity(3, 3) * 0;
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
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;

  return settings;
}
