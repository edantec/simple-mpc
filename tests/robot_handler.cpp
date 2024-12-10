
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
  {
    BOOST_CHECK(model_handler.getCompleteModel() == model);
    BOOST_CHECK_EQUAL(model_handler.getModel().nq, 17);
    BOOST_CHECK_EQUAL(model_handler.getModel().nv, 16);
  }

  // Base frame
  {
    BOOST_CHECK_EQUAL(model.frames.at(model_handler.getBaseFrameId()).name, base_frame);
  }

  // Feet tests
  {
    for(size_t i=0; i < feet_names.size(); i++)
    {
      const std::string foot_name = feet_names.at(i);
      BOOST_CHECK_EQUAL(foot_name, model_handler.getFootName(i));
      BOOST_CHECK_EQUAL(foot_name, model_handler.getFeetNames().at(i));
      BOOST_CHECK_EQUAL(foot_name, model.frames.at(model_handler.getFeetIds().at(i)).name);
      BOOST_CHECK_EQUAL(foot_name, model.frames.at(model_handler.getFootId(foot_name)).name);

      const FrameIndex ref_frame = model_handler.getRefFootId(foot_name);
      const FrameIndex ref_frame_parent = model_handler.getModel().frames.at(ref_frame).parentFrame;
      BOOST_CHECK_EQUAL(model_handler.getModel().frames.at(ref_frame_parent).name, base_frame);
      BOOST_CHECK(model_handler.getModel().frames.at(ref_frame).placement.isApprox(feet_refs.at(i)));
    }
  }

  // State
  {
    const size_t nq_red = 17;
    Eigen::Vector<double, 19> q = pinocchio::randomConfiguration(model); q.head<3>() = Eigen::Vector3d::Random();
    Eigen::Vector<double, 18> v = Eigen::Vector<double, 18>::Random();
    Eigen::Vector<double, 33> x = Eigen::Vector<double, 33>::Random();
    bool is_reference_state = false;

    for(int n=0;n<2;n++) // First time with random data, second with reference state
    {
      // State vector without locked joints
      for(size_t i = 1; i< model_handler.getModel().njoints; i++) {
        const std::string& joint_name = model_handler.getModel().names[i];
        const JointModel& joint_full = model.joints[model.getJointId(joint_name)];
        const JointModel& joint_red = model_handler.getModel().joints[i];

        x.segment(joint_red.idx_q(), joint_red.nq()) = q.segment(joint_full.idx_q(), joint_full.nq());
        x.segment(nq_red + joint_red.idx_v(), joint_red.nv()) = v.segment(joint_full.idx_v(), joint_full.nv());
      }

      // Test shape state
      BOOST_CHECK(model_handler.shapeState(q, v).isApprox(x));

      // Test reference state
      if(is_reference_state) {
        BOOST_CHECK(x.isApprox(model_handler.getReferenceState()));
      }

      // Set reference state for 2nd round
      q = model.referenceConfigurations[default_conf_name];
      v = Eigen::Vector<double, 18>::Zero();
      is_reference_state = true;
    }
  }

  // Difference
  {
  // Eigen::Vector<double, 33> x1 = Eigen::Vector<double, 33>::Random();
  // Eigen::Vector<double, 33> x2 = Eigen::Vector<double, 33>::Random();
  }
  // Eigen::VectorXd difference(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const;

  BOOST_CHECK_EQUAL(model_handler.getMass(), 90.272192000000018);
}

BOOST_AUTO_TEST_CASE(build_solo) {
  // RobotDataHandler handler = getSoloHandler();

  // BOOST_CHECK_EQUAL(handler.getModel().nq, 19);
  // BOOST_CHECK_EQUAL(handler.getModel().nv, 18);
  // BOOST_CHECK_EQUAL(handler.getMass(), 2.5000027900000004);

  // Eigen::VectorXd q1(19);
  // q1.setZero();
  // q1[6] = 1;

  // handler.updateConfiguration(q1, false);
  // BOOST_CHECK_EQUAL(handler.getConfiguration(), q1);
  // BOOST_CHECK_EQUAL(handler.getFootName(1), "FL_FOOT");

  // Eigen::Vector3d com = handler.getComPosition();
  // pinocchio::SE3 pose = handler.getFootPose("FL_FOOT");
}

BOOST_AUTO_TEST_SUITE_END()
