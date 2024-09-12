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

  BOOST_CHECK_EQUAL(fdproblem.getCostMap().at("control_cost"), 1);
  BOOST_CHECK_EQUAL(fdproblem.getCostMap().at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(fdproblem.getCostMap().at("left_sole_link_pose_cost"), 3);

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
  StageModel sm = fdproblem.createStage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 7);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 3);

  fdproblem.createProblem(settings.x0, 100, 6, settings.gravity[2]);

  BOOST_CHECK_EQUAL(fdproblem.getProblem()->stages_.size(), 100);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  fdproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(4, "left_sole_link"),
                    pose_left_random);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  fdproblem.setReferencePoses(3, new_poses);
  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  fdproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  fdproblem.setReferenceForce(5, "left_sole_link",
                              force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));
}

BOOST_AUTO_TEST_CASE(kinodynamics) {
  RobotHandler handler = getTalosHandler();
  KinodynamicsSettings settings = getKinodynamicsSettings(handler);

  KinodynamicsProblem knproblem(settings, handler);

  BOOST_CHECK_EQUAL(knproblem.getCostMap().at("control_cost"), 1);
  BOOST_CHECK_EQUAL(knproblem.getCostMap().at("centroidal_cost"), 2);
  BOOST_CHECK_EQUAL(knproblem.getCostMap().at("left_sole_link_pose_cost"), 4);

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
  StageModel sm = knproblem.createStage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  knproblem.createProblem(settings.x0, 100, 6, settings.gravity[2]);

  BOOST_CHECK_EQUAL(knproblem.getProblem()->stages_.size(), 100);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  knproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(knproblem.getReferencePose(4, "left_sole_link"),
                    pose_left_random);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  knproblem.setReferencePoses(3, new_poses);

  BOOST_CHECK_EQUAL(knproblem.getReferencePose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferencePose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  knproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  knproblem.setReferenceForce(5, "left_sole_link",
                              force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));
}

BOOST_AUTO_TEST_CASE(centroidal) {
  RobotHandler handler = getTalosHandler();
  CentroidalSettings settings = getCentroidalSettings(handler);

  CentroidalProblem cproblem(settings, handler);

  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("control_cost"), 0);
  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("linear_mom_cost"), 1);
  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("angular_acc_cost"), 4);

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
  StageModel sm = cproblem.createStage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 5);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  cproblem.createProblem(settings.x0, 100, 6, settings.gravity[2]);

  BOOST_CHECK_EQUAL(cproblem.getProblem()->stages_.size(), 100);

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  cproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "left_sole_link"),
                    force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "right_sole_link"),
                    force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  cproblem.setReferenceForce(5, "left_sole_link",
                             force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(5, "left_sole_link"),
                    force_refs.at("left_sole_link"));

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  cproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(
      cproblem.getReferencePose(4, "left_sole_link").translation(),
      pose_left_random.translation());

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  cproblem.setReferencePoses(3, new_poses);

  BOOST_CHECK_EQUAL(cproblem.getReferencePose(3, "left_sole_link"),
                    new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferencePose(3, "right_sole_link"),
                    new_poses.at("right_sole_link"));
}

BOOST_AUTO_TEST_CASE(centroidal_solo) {
  RobotHandler handler = getSoloHandler();
  CentroidalSettings settings = getCentroidalSettings(handler);
  settings.force_size = 3;

  CentroidalProblem cproblem(settings, handler);

  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("control_cost"), 0);
  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("linear_mom_cost"), 1);
  BOOST_CHECK_EQUAL(cproblem.getCostMap().at("angular_acc_cost"), 4);

  std::vector<std::string> contact_names = {"FR_FOOT", "FL_FOOT", "HR_FOOT",
                                            "HL_FOOT"};
  std::vector<bool> contact_states = {true, true, true, false};
  StdVectorEigenAligned<Eigen::Vector3d> contact_poses;
  Eigen::Vector3d p1 = handler.getFootPose("FR_FOOT").translation();
  Eigen::Vector3d p2 = handler.getFootPose("FL_FOOT").translation();
  Eigen::Vector3d p3 = handler.getFootPose("HR_FOOT").translation();
  Eigen::Vector3d p4 = handler.getFootPose("HL_FOOT").translation();
  contact_poses.push_back(p1);
  contact_poses.push_back(p2);
  contact_poses.push_back(p3);
  contact_poses.push_back(p4);
  ContactMap cm(contact_names, contact_states, contact_poses);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(3);
  f1 << 0, 0, handler.getMass() / 3;
  force_refs.insert({"FR_FOOT", f1});
  force_refs.insert({"FL_FOOT", f1});
  force_refs.insert({"HR_FOOT", f1});
  force_refs.insert({"HL_FOOT", Eigen::VectorXd::Zero(3)});
  StageModel sm = cproblem.createStage(cm, force_refs);
  CostStack *cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 5);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 0);

  cproblem.createProblem(settings.x0, 100, 3, settings.gravity[2]);

  BOOST_CHECK_EQUAL(cproblem.getProblem()->stages_.size(), 100);

  force_refs.at("FR_FOOT")[1] = 1;
  force_refs.at("FL_FOOT")[0] = 1;
  cproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "FR_FOOT"),
                    force_refs.at("FR_FOOT"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "FL_FOOT"),
                    force_refs.at("FL_FOOT"));
}

BOOST_AUTO_TEST_SUITE_END()
