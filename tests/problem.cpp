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

  BOOST_CHECK_EQUAL(fdproblem.get_cost_map().at("control_cost"), 1);
  BOOST_CHECK_EQUAL(fdproblem.get_cost_map().at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(fdproblem.get_cost_map().at("left_sole_link_pose_cost"), 3);

  std::vector<std::string> contact_names = {"left_sole_link",
                                            "right_sole_link"};
  std::vector<bool> contact_states = {true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_names, contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = fdproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 7);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 3);

  fdproblem.create_problem(settings.x0, 100, 6, settings.gravity[2]);
  BOOST_CHECK_EQUAL(fdproblem.get_problem()->stages_.size(), 100);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  fdproblem.set_reference_pose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(fdproblem.get_reference_pose(4, "left_sole_link"),
                    pose_left_random);

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

  force_refs.at("left_sole_link")[2] = 250;
  fdproblem.set_reference_force(5, "left_sole_link",
                                force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.get_reference_force(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));
}

BOOST_AUTO_TEST_CASE(kinodynamics) {
  RobotHandler handler = getTalosHandler();
  KinodynamicsSettings settings = getKinodynamicsSettings(handler);

  KinodynamicsProblem knproblem(settings, handler);

  BOOST_CHECK_EQUAL(knproblem.get_cost_map().at("control_cost"), 1);
  BOOST_CHECK_EQUAL(knproblem.get_cost_map().at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(knproblem.get_cost_map().at("left_sole_link_pose_cost"), 4);

  std::vector<std::string> contact_names = {"left_sole_link",
                                            "right_sole_link"};
  std::vector<bool> contact_states = {true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_names, contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = knproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  knproblem.create_problem(settings.x0, 100, 6, settings.gravity[2]);

  BOOST_CHECK_EQUAL(knproblem.get_problem()->stages_.size(), 100);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  knproblem.set_reference_pose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(knproblem.get_reference_pose(4, "left_sole_link"),
                    pose_left_random);

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

  force_refs.at("left_sole_link")[2] = 250;
  knproblem.set_reference_force(5, "left_sole_link",
                                force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.get_reference_force(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));
}

BOOST_AUTO_TEST_CASE(centroidal) {
  RobotHandler handler = getTalosHandler();
  CentroidalSettings settings = getCentroidalSettings(handler);

  CentroidalProblem cproblem(settings, handler);

  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("control_cost"), 0);
  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("linear_mom_cost"), 1);
  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("angular_acc_cost"), 4);

  std::vector<bool> contact_states = {true, false};
  std::vector<std::string> contact_names = {"left_sole_link",
                                            "right_sole_link"};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = {0, 0.1, 0};
  Eigen::Vector3d p2 = {0, -0.1, 0};
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  ContactMap cm(contact_names, contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = cproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 5);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  cproblem.create_problem(settings.x0, 100, 6, settings.gravity[2]);

  BOOST_CHECK_EQUAL(cproblem.get_problem()->stages_.size(), 100);

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  cproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  cproblem.set_reference_force(5, "left_sole_link",
                               force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  cproblem.set_reference_pose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(
      cproblem.get_reference_pose(4, "left_sole_link").translation(),
      pose_left_random.translation());

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  cproblem.set_reference_poses(3, new_poses);

  BOOST_CHECK_EQUAL(cproblem.get_reference_pose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.get_reference_pose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));
}

BOOST_AUTO_TEST_CASE(centroidal_solo) {
  RobotHandler handler = getSoloHandler();
  CentroidalSettings settings = getCentroidalSettings(handler);
  settings.force_size = 3;

  CentroidalProblem cproblem(settings, handler);

  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("control_cost"), 0);
  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("linear_mom_cost"), 1);
  BOOST_CHECK_EQUAL(cproblem.get_cost_map().at("angular_acc_cost"), 4);

  std::vector<std::string> contact_names = {"FR_FOOT", "FL_FOOT", "HR_FOOT",
                                            "HL_FOOT"};
  std::vector<bool> contact_states = {true, true, true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = handler.get_ee_pose("FR_FOOT").translation();
  Eigen::Vector3d p2 = handler.get_ee_pose("FL_FOOT").translation();
  Eigen::Vector3d p3 = handler.get_ee_pose("HR_FOOT").translation();
  Eigen::Vector3d p4 = handler.get_ee_pose("HL_FOOT").translation();
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  contact_poses.push_back(p3);
  contact_poses.push_back(p4);
  ContactMap cm(contact_names, contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(3);
  f1 << 0, 0, handler.get_mass() / 3;
  force_refs.insert({"FR_FOOT", f1});
  force_refs.insert({"FL_FOOT", f1});
  force_refs.insert({"HR_FOOT", f1});
  force_refs.insert({"HL_FOOT", Eigen::VectorXd::Zero(3)});
  StageModel sm = cproblem.create_stage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 5);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  cproblem.create_problem(settings.x0, 100, 3, settings.gravity[2]);

  BOOST_CHECK_EQUAL(cproblem.get_problem()->stages_.size(), 100);

  force_refs.at("FR_FOOT")[1] = 1;
  force_refs.at("FL_FOOT")[0] = 1;
  cproblem.set_reference_forces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "FR_FOOT"),
                    force_refs.at("FR_FOOT"));
  BOOST_CHECK_EQUAL(cproblem.get_reference_force(3, "FL_FOOT"),
                    force_refs.at("FL_FOOT"));
}

BOOST_AUTO_TEST_SUITE_END()
