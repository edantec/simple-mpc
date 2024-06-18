
#include <boost/test/unit_test.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/mpc.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(mpc)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(mpc_fulldynamics) {
  RobotHandler handler = getTalosHandler();

  FullDynamicsSettings settings = getFullDynamicsSettings(handler);
  FullDynamicsProblem fdproblem(settings, handler);

  double mass = handler.get_mass();
  std::vector<ContactMap> initial_contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> initial_force_sequence;
  std::size_t T = 100;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, mass * settings.gravity[2] / 2., 0, 0, 0;
  std::map<std::string, Eigen::VectorXd> force_refs;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", f1});
  for (std::size_t i = 0; i < T; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    initial_contact_sequence.push_back(cm1);
    initial_force_sequence.push_back(force_refs);
  }

  fdproblem.create_problem(settings.x0, initial_contact_sequence,
                           initial_force_sequence);

  std::shared_ptr<Problem> problem =
      std::make_shared<FullDynamicsProblem>(fdproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.T = T;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = 1000;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(handler.get_rmodel().nv - 6);
  u0.setZero();

  MPC mpc = MPC(mpc_settings, problem, settings.x0, u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  Eigen::VectorXd f2(6);
  f2 << 0, 0, mass * settings.gravity[2], 0, 0, 0;
  Eigen::VectorXd f0 = Eigen::VectorXd::Zero(6);
  std::map<std::string, Eigen::VectorXd> force_refs_left;
  force_refs_left.insert({"left_sole_link", f2});
  force_refs_left.insert({"right_sole_link", f0});
  std::map<std::string, Eigen::VectorXd> force_refs_right;
  force_refs_right.insert({"left_sole_link", f0});
  force_refs_right.insert({"right_sole_link", f2});
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_left);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {false, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_right);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }

  mpc.generateFullHorizon(contact_sequence, force_sequence);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  for (std::size_t i = 0; i < 50; i++)
    mpc.recedeWithCycle();

  BOOST_CHECK_EQUAL(
      mpc.get_problem()->get_reference_force(80, "right_sole_link"), f0);
}

BOOST_AUTO_TEST_CASE(mpc_kinodynamics) {
  RobotHandler handler = getTalosHandler();

  KinodynamicsSettings settings = getKinodynamicsSettings(handler);
  KinodynamicsProblem kinoproblem(settings, handler);

  double mass = handler.get_mass();
  std::vector<ContactMap> initial_contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> initial_force_sequence;
  std::size_t T = 100;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, mass * settings.gravity[2] / 2., 0, 0, 0;
  std::map<std::string, Eigen::VectorXd> force_refs;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", f1});
  for (std::size_t i = 0; i < T; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    initial_contact_sequence.push_back(cm1);
    initial_force_sequence.push_back(force_refs);
  }

  kinoproblem.create_problem(settings.x0, initial_contact_sequence,
                             initial_force_sequence);

  std::shared_ptr<Problem> problem =
      std::make_shared<KinodynamicsProblem>(kinoproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.T = T;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = 1000;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(handler.get_rmodel().nv + 6);
  u0.setZero();
  u0.head(12) << f1, f1;

  MPC mpc = MPC(mpc_settings, problem, settings.x0, u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  Eigen::VectorXd f2(6);
  f2 << 0, 0, mass * settings.gravity[2], 0, 0, 0;
  Eigen::VectorXd f0 = Eigen::VectorXd::Zero(6);
  std::map<std::string, Eigen::VectorXd> force_refs_left;
  force_refs_left.insert({"left_sole_link", f2});
  force_refs_left.insert({"right_sole_link", f0});
  std::map<std::string, Eigen::VectorXd> force_refs_right;
  force_refs_right.insert({"left_sole_link", f0});
  force_refs_right.insert({"right_sole_link", f2});
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_left);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {false, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_right);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }

  mpc.generateFullHorizon(contact_sequence, force_sequence);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  for (std::size_t i = 0; i < 50; i++)
    mpc.recedeWithCycle();

  BOOST_CHECK_EQUAL(
      mpc.get_problem()->get_reference_force(80, "right_sole_link"), f0);
}

BOOST_AUTO_TEST_CASE(mpc_centroidal) {
  RobotHandler handler = getTalosHandler();

  CentroidalSettings settings = getCentroidalSettings(handler);
  CentroidalProblem centproblem(settings, handler);

  double mass = handler.get_mass();
  std::vector<ContactMap> initial_contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> initial_force_sequence;
  std::size_t T = 100;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, mass * settings.gravity[2] / 2., 0, 0, 0;
  std::map<std::string, Eigen::VectorXd> force_refs;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", f1});
  for (std::size_t i = 0; i < T; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    initial_contact_sequence.push_back(cm1);
    initial_force_sequence.push_back(force_refs);
  }

  centproblem.create_problem(settings.x0, initial_contact_sequence,
                             initial_force_sequence);
  std::shared_ptr<Problem> problem =
      std::make_shared<CentroidalProblem>(centproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.T = T;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = 1000;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(12);
  u0.setZero();
  u0.head(12) << f1, f1;

  MPC mpc = MPC(mpc_settings, problem, handler.get_x0(), u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<ContactMap> contact_sequence;
  std::vector<std::map<std::string, Eigen::VectorXd>> force_sequence;
  Eigen::VectorXd f2(6);
  f2 << 0, 0, mass * settings.gravity[2], 0, 0, 0;
  Eigen::VectorXd f0 = Eigen::VectorXd::Zero(6);
  std::map<std::string, Eigen::VectorXd> force_refs_left;
  force_refs_left.insert({"left_sole_link", f2});
  force_refs_left.insert({"right_sole_link", f0});
  std::map<std::string, Eigen::VectorXd> force_refs_right;
  force_refs_right.insert({"left_sole_link", f0});
  force_refs_right.insert({"right_sole_link", f2});
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {true, false};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_left);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::vector<bool> contact_states = {false, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs_right);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::vector<bool> contact_states = {true, true};
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses = {
        handler.get_ee_pose(0).translation(),
        handler.get_ee_pose(1).translation()};
    ContactMap cm1(contact_states, contact_poses);
    contact_sequence.push_back(cm1);
    force_sequence.push_back(force_refs);
  }

  mpc.generateFullHorizon(contact_sequence, force_sequence);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  for (std::size_t i = 0; i < 50; i++)
    mpc.recedeWithCycle();

  BOOST_CHECK_EQUAL(
      mpc.get_problem()->get_reference_force(80, "right_sole_link"), f0);
}

BOOST_AUTO_TEST_SUITE_END()
