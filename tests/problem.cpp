
#include <boost/test/unit_test.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/robot-handler.hpp"

BOOST_AUTO_TEST_SUITE(problem)

using namespace simple_mpc;
using T = double;
using context::MatrixXs;
using context::VectorXs;
using QuadraticResidualCost = QuadraticResidualCostTpl<T>;

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

BOOST_AUTO_TEST_CASE(fulldynamics) {
  RobotHandler handler = getTalosHandler();
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

  FullDynamicsProblem fdproblem(settings, handler);

  BOOST_CHECK_EQUAL(fdproblem.cost_map_.at("control_cost"), 1);
  BOOST_CHECK_EQUAL(fdproblem.cost_map_.at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(fdproblem.cost_map_.at("left_sole_link_pose_cost"), 3);

  std::vector<bool> contact_states = {true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_states, contact_poses);

  std::vector<Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.push_back(f1);
  force_refs.push_back(Eigen::VectorXd::Zero(6));
  StageModel sm = fdproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 7);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 2);

  std::vector<ContactMap> contact_sequence;
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0.5, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
  }
  fdproblem.create_problem(settings.x0, contact_sequence);
  BOOST_CHECK_EQUAL(fdproblem.problem_->stages_.size(), 70);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::vector<pinocchio::SE3> new_poses = {new_pose_left, new_pose_right};

  fdproblem.set_reference_poses(3, new_poses);
  BOOST_CHECK_EQUAL(fdproblem.get_reference_pose(3, "left_sole_link_pose_cost"),
                    new_poses[0]);
  BOOST_CHECK_EQUAL(
      fdproblem.get_reference_pose(3, "right_sole_link_pose_cost"),
      new_poses[1]);

  force_refs[0][1] = 1;
  force_refs[1][0] = 1;
  fdproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(
      fdproblem.get_reference_force(3, "left_sole_link_force_cost"),
      force_refs[0]);
  BOOST_CHECK_EQUAL(
      fdproblem.get_reference_force(3, "right_sole_link_force_cost"),
      force_refs[1]);
}

BOOST_AUTO_TEST_SUITE_END()
