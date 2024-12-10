
#include <boost/test/unit_test.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"
#include <pinocchio/fwd.hpp>


BOOST_AUTO_TEST_SUITE(robot_handler)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(model_handler) {
  /***************/
  /* Test values */
  /***************/
  Model model;
  const std::string base_frame = "root_joint";
  const std::string default_conf_name = "straight_standing";
  const std::vector<std::string> locked_joints { "FR_HFE", "FL_HFE"}; // Lock two random joint for testing
  const std::vector<std::string> feet_names { "FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"};
  const std::vector<SE3> feet_refs {
    SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d( 0.1, -0.1, 0.)),
    SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d( 0.1,  0.1, 0.)),
    SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(-0.1, -0.1, 0.)),
    SE3(Eigen::Quaternion(0.,0.,0.,1.), Eigen::Vector3d(-0.1,  0.1, 0.))
  };

  /************************/
  /* Create model handler */
  /************************/
  // Load pinocchio model from example robot data
  const std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
  const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf";

  pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
  srdf::loadReferenceConfigurations(model, srdf_path, false);

  RobotModelHandler model_handler(model, default_conf_name, base_frame, locked_joints);

  // Add feet
  for(size_t i = 0; i < feet_names.size(); i++)
  {
    model_handler.addFoot(feet_names.at(i), base_frame, feet_refs.at(i));
  }

  /*********/
  /* Tests */
  /*********/
  // Model
  BOOST_CHECK(model_handler.getCompleteModel() == model);
  BOOST_CHECK_EQUAL(model_handler.getModel().nq, 17);
  BOOST_CHECK_EQUAL(model_handler.getModel().nv, 16);

  // Base frame
  BOOST_CHECK_EQUAL(model.frames.at(model_handler.getBaseFrameId()).name, base_frame);

  // Feet tests
  for(size_t i=0; i < feet_names.size(); i++)
  {
    const std::string foot_name = feet_names.at(i);
    BOOST_CHECK_EQUAL(foot_name, model_handler.getFootName(i));
    BOOST_CHECK_EQUAL(foot_name, model_handler.getFeetNames().at(i));
    BOOST_CHECK_EQUAL(foot_name, model.frames.at(model_handler.getFeetIds().at(i)).name);
    BOOST_CHECK_EQUAL(foot_name, model.frames.at(model_handler.getFootId(foot_name)).name);

    const FrameIndex ref_frame = model_handler.getRefFootId(foot_name);
    BOOST_CHECK_EQUAL(model.frames.at(ref_frame).parentFrame, model.getFrameId(base_frame));
    BOOST_CHECK(model.frames.at(ref_frame).placement.isApprox(feet_refs.at(i)));
  }

  // // State
  // Eigen::VectorXd difference(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const;
  // Eigen::VectorXd shapeState(const Eigen::VectorXd &q, const Eigen::VectorXd &v) const;
  // const Eigen::VectorXd& getReferenceState() const


  BOOST_CHECK_EQUAL(model_handler.getMass(), 90.272192000000018);
}

BOOST_AUTO_TEST_CASE(build_solo) {
  RobotDataHandler handler = getSoloHandler();

  BOOST_CHECK_EQUAL(handler.getModel().nq, 19);
  BOOST_CHECK_EQUAL(handler.getModel().nv, 18);
  BOOST_CHECK_EQUAL(handler.getMass(), 2.5000027900000004);

  Eigen::VectorXd q1(19);
  q1.setZero();
  q1[6] = 1;

  handler.updateConfiguration(q1, false);
  BOOST_CHECK_EQUAL(handler.getConfiguration(), q1);
  BOOST_CHECK_EQUAL(handler.getFootName(1), "FL_FOOT");

  Eigen::Vector3d com = handler.getComPosition();
  pinocchio::SE3 pose = handler.getFootPose("FL_FOOT");
}

BOOST_AUTO_TEST_SUITE_END()
