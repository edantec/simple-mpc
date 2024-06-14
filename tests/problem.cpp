
#include <boost/test/unit_test.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(problem)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(fulldynamics) {
  RobotHandler handler = getTalosHandler();

  FullDynamicsSettings settings = getFullDynamicsSettings(handler);

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

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = fdproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 7);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 2);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0.5, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  fdproblem.create_problem(settings.x0, contact_sequence, force_sequence);
  BOOST_CHECK_EQUAL(fdproblem.problem_->stages_.size(), 70);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  fdproblem.set_reference_poses(3, new_poses);
  BOOST_CHECK_EQUAL(fdproblem.get_reference_pose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.get_reference_pose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  fdproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(fdproblem.get_reference_force(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.get_reference_force(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));
}

BOOST_AUTO_TEST_CASE(kinodynamics) {
  RobotHandler handler = getTalosHandler();
  int nv = handler.get_rmodel().nv;
  int nu = nv + 6;

  KinodynamicsSettings settings;
  settings.x0 = handler.get_x0();
  settings.u0 = Eigen::VectorXd::Zero(nu);
  settings.DT = 0.01;
  settings.w_x = Eigen::MatrixXd::Identity(nv * 2, nv * 2);
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu);
  settings.w_cent = Eigen::MatrixXd::Identity(6, 6);
  settings.w_centder = Eigen::MatrixXd::Identity(6, 6);
  settings.gravity << 0, 0, 9;
  settings.force_size = 6;
  settings.w_frame = Eigen::MatrixXd::Identity(6, 6);
  settings.qmin = -Eigen::VectorXd::Ones(handler.get_rmodel().nv);
  settings.qmax = Eigen::VectorXd::Ones(handler.get_rmodel().nv);

  KinodynamicsProblem knproblem(settings, handler);

  BOOST_CHECK_EQUAL(knproblem.cost_map_.at("control_cost"), 1);
  BOOST_CHECK_EQUAL(knproblem.cost_map_.at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(knproblem.cost_map_.at("left_sole_link_pose_cost"), 4);

  std::vector<bool> contact_states = {true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = knproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0.5, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }

  knproblem.create_problem(settings.x0, contact_sequence, force_sequence);

  BOOST_CHECK_EQUAL(knproblem.problem_->stages_.size(), 70);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  knproblem.set_reference_poses(3, new_poses);

  BOOST_CHECK_EQUAL(knproblem.get_reference_pose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.get_reference_pose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  knproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(knproblem.get_reference_force(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.get_reference_force(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));
}

BOOST_AUTO_TEST_CASE(centroidal) {
  RobotHandler handler = getTalosHandler();
  int nx = 18;
  int nu = 6 * 2;

  CentroidalSettings settings;
  Eigen::VectorXd x0(nx);
  x0 << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  settings.x0 = x0;
  settings.u0 = Eigen::VectorXd::Zero(nu);
  settings.DT = 0.01;
  settings.w_x = Eigen::MatrixXd::Identity(nx, nx);
  settings.w_u = Eigen::MatrixXd::Identity(nu, nu);
  settings.w_linear_mom = Eigen::MatrixXd::Identity(3, 3);
  settings.w_angular_mom = Eigen::MatrixXd::Identity(3, 3);
  settings.w_linear_acc = Eigen::MatrixXd::Identity(3, 3);
  settings.w_angular_acc = Eigen::MatrixXd::Identity(3, 3);
  settings.gravity << 0, 0, 9;
  settings.force_size = 6;

  CentroidalProblem cproblem(settings, handler);

  BOOST_CHECK_EQUAL(cproblem.cost_map_.at("control_cost"), 0);
  BOOST_CHECK_EQUAL(cproblem.cost_map_.at("linear_mom_cost"), 1);
  BOOST_CHECK_EQUAL(cproblem.cost_map_.at("angular_acc_cost"), 4);

  std::vector<bool> contact_states = {true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = cproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 5);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {{0, 0.1, 0},
                                                            {0.5, -0.1, 0}};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }

  cproblem.create_problem(settings.x0, contact_sequence, force_sequence);

  BOOST_CHECK_EQUAL(cproblem.problem_->stages_.size(), 70);

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  cproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));
}

BOOST_AUTO_TEST_SUITE_END()
