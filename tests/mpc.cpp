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

  size_t T = 100;
  fdproblem.create_problem(settings.x0, T, 6, -settings.gravity[2]);

  std::shared_ptr<Problem> problem =
      std::make_shared<FullDynamicsProblem>(fdproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = -handler.get_mass() * settings.gravity[2];

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(handler.get_rmodel().nv - 6);
  u0.setZero();

  MPC mpc = MPC(mpc_settings, problem, settings.x0, u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  // std::vector<std::vector<bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), false});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateFullHorizon(contact_states);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  for (std::size_t i = 0; i < 10; i++) {
    mpc.iterate(settings.x0.head(handler.get_rmodel().nq),
                settings.x0.tail(handler.get_rmodel().nv));
  }

  BOOST_CHECK_EQUAL(mpc.get_horizon_iteration(), 10);
}

BOOST_AUTO_TEST_CASE(mpc_kinodynamics) {
  RobotHandler handler = getTalosHandler();

  KinodynamicsSettings settings = getKinodynamicsSettings(handler);
  KinodynamicsProblem kinoproblem(settings, handler);
  std::size_t T = 100;
  double support_force = -handler.get_mass() * settings.gravity[2];
  Eigen::VectorXd f1(6);
  f1 << 0, 0, support_force, 0, 0, 0;

  kinoproblem.create_problem(settings.x0, T, 6, -settings.gravity[2]);

  std::shared_ptr<Problem> problem =
      std::make_shared<KinodynamicsProblem>(kinoproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = support_force;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(handler.get_rmodel().nv + 6);
  u0.setZero();
  u0.head(12) << f1, f1;

  MPC mpc = MPC(mpc_settings, problem, settings.x0, u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  // std::vector<std::vector<bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), false});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateFullHorizon(contact_states);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  for (std::size_t i = 0; i < 10; i++) {
    mpc.iterate(settings.x0.head(handler.get_rmodel().nq),
                settings.x0.tail(handler.get_rmodel().nv));
  }

  BOOST_CHECK_EQUAL(mpc.get_horizon_iteration(), 10);
}

BOOST_AUTO_TEST_CASE(mpc_centroidal) {
  RobotHandler handler = getTalosHandler();

  CentroidalSettings settings = getCentroidalSettings(handler);
  CentroidalProblem centproblem(settings, handler);

  std::vector<std::string> contact_names = {"left_sole_link",
                                            "right_sole_link"};
  double support_force = -handler.get_mass() * settings.gravity[2];
  std::size_t T = 100;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, support_force / 2., 0, 0, 0;

  centproblem.create_problem(settings.x0, T, 6, -settings.gravity[2]);
  std::shared_ptr<Problem> problem =
      std::make_shared<CentroidalProblem>(centproblem);

  MPCSettings mpc_settings;
  mpc_settings.totalSteps = 4;
  mpc_settings.ddpIteration = 1;

  mpc_settings.min_force = 150;
  mpc_settings.support_force = support_force;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;

  mpc_settings.num_threads = 8;
  Eigen::VectorXd u0(12);
  u0.setZero();
  u0.head(12) << f1, f1;

  MPC mpc = MPC(mpc_settings, problem, handler.get_x0(), u0);

  BOOST_CHECK_EQUAL(mpc.get_xs().size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.get_us().size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  // std::vector<std::vector<bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), false});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++) {
    std::map<std::string, bool> contact_state;
    contact_state.insert({handler.get_ee_name(0), true});
    contact_state.insert({handler.get_ee_name(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateFullHorizon(contact_states);

  BOOST_CHECK_EQUAL(mpc.get_fullHorizon().size(), 130);
  BOOST_CHECK_EQUAL(mpc.get_fullHorizonData().size(), 130);

  /* for (std::size_t i = 0; i < 50; i++)
    mpc.recedeWithCycle();

  BOOST_CHECK_EQUAL(
      mpc.get_problem()->get_reference_force(80, "right_sole_link"), f0); */

  for (std::size_t i = 0; i < 10; i++) {
    mpc.iterate(handler.get_x0().head(handler.get_rmodel().nq),
                handler.get_x0().tail(handler.get_rmodel().nv));
  }

  BOOST_CHECK_EQUAL(mpc.get_horizon_iteration(), 10);
}

BOOST_AUTO_TEST_SUITE_END()
