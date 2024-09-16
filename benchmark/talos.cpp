#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/mpc.hpp"
#include "simple-mpc/robot-handler.hpp"

using simple_mpc::ContactMap;
using simple_mpc::RobotHandler;
using simple_mpc::RobotHandlerSettings;
using PoseVec = aligator::StdVectorEigenAligned<Eigen::Vector3d>;
using simple_mpc::FullDynamicsProblem;
using simple_mpc::FullDynamicsSettings;
using simple_mpc::MPC;
using simple_mpc::MPCSettings;
using simple_mpc::Problem;

int main() {
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
  settings.end_effector_names = {
      "left_sole_link",
      "right_sole_link",
  };
  settings.root_name = "root_joint";
  settings.base_configuration = "half_sitting";

  RobotHandler handler = RobotHandler();
  handler.initialize(settings);

  size_t T = 100;

  FullDynamicsSettings problem_settings;
  int nu = handler.getModel().nv - 6;
  int ndx = handler.getModel().nv * 2;

  Eigen::VectorXd w_x_vec(ndx);
  w_x_vec << 0, 0, 0, 100, 100, 100,  // Base pos/ori
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   // Left leg
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   // Right leg
      10, 10,                         // Torso
      1, 1, 1, 1,                     // Left arm
      1, 1, 1, 1,                     // Right arm
      1, 1, 1, 1, 1, 1,               // Base pos/ori vel
      0.1, 0.1, 0.1, 0.1, 0.01, 0.01, // Left leg vel
      0.1, 0.1, 0.1, 0.1, 0.01, 0.01, // Right leg vel
      10, 10,                         // Torso vel
      1, 1, 1, 1,                     // Left arm vel
      1, 1, 1, 1;                     // Right arm vel
  Eigen::VectorXd w_cent(6);
  w_cent << 0, 0, 0, 0.1, 0., 100;
  Eigen::VectorXd w_forces(6);
  w_forces << 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01;

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);

  problem_settings.x0 = handler.getState();
  problem_settings.u0 = u0;
  problem_settings.DT = 0.01;
  problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
  problem_settings.w_x.diagonal() = w_x_vec;
  problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
  problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
  problem_settings.w_cent.diagonal() = w_cent;
  problem_settings.gravity = {0, 0, -9.81};
  problem_settings.force_size = 6,
  problem_settings.w_forces = Eigen::MatrixXd::Zero(6, 6);
  problem_settings.w_forces.diagonal() = w_forces;
  problem_settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 2000;
  problem_settings.umin = -handler.getModel().effortLimit.tail(nu);
  problem_settings.umax = handler.getModel().effortLimit.tail(nu);
  problem_settings.qmin = handler.getModel().lowerPositionLimit.tail(nu);
  problem_settings.qmax = handler.getModel().upperPositionLimit.tail(nu);
  problem_settings.mu = 0.8;
  problem_settings.Lfoot = 0.1;
  problem_settings.Wfoot = 0.075;

  FullDynamicsProblem problem = FullDynamicsProblem(handler);
  problem.initialize(problem_settings);
  problem.createProblem(handler.getState(), T, 6, problem_settings.gravity[2]);

  std::shared_ptr<Problem> problemPtr =
      std::make_shared<FullDynamicsProblem>(problem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.min_force = 150;
  mpc_settings.support_force = -problem_settings.gravity[2] * handler.getMass();
  mpc_settings.TOL = 1e-4;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.max_iters = 1;
  mpc_settings.num_threads = 2;

  MPC mpc = MPC(mpc_settings, problemPtr);

  std::vector<std::map<std::string, bool>> contact_states;
  // std::vector<std::vector<bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.getFootName(0), true});
    contact_state.insert({handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.getFootName(0), true});
    contact_state.insert({handler.getFootName(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.getFootName(0), true});
    contact_state.insert({handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.getFootName(0), false});
    contact_state.insert({handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.getFootName(0), true});
    contact_state.insert({handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateFullHorizon(contact_states);

  return 0;
}
