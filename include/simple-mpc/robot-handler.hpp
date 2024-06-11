#ifndef SIMPLE_MPC_ROBOT_HANDLER
#define SIMPLE_MPC_ROBOT_HANDLER

#include <pinocchio/fwd.hpp>
// Include pinocchio first
#include <Eigen/Dense>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <vector>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {

struct RobotHandlerSettings {
public:
  std::string urdf_path = "";
  std::string srdf_path = "";
  std::string robot_description = "";
  std::vector<std::string> controlled_joints_names;

  std::vector<std::string> end_effector_names;
  std::string root_name = "";
  std::string base_configuration = "";
};

class RobotHandler {
private:
  RobotHandlerSettings settings_;

  std::vector<unsigned long> controlled_joints_id_;
  std::vector<unsigned long> end_effector_ids_;
  unsigned long root_ids_;

  pinocchio::Model rmodel_complete_, rmodel_;
  pinocchio::Data rdata_;
  // std::vector<pinocchio::JointIndex> pinocchioControlledJoints_;

  Eigen::VectorXd q0Complete_, q0_;
  Eigen::VectorXd v0Complete_, v0_;
  Eigen::VectorXd x0_;
  Eigen::VectorXd x_internal_;

  Eigen::Vector3d com_position_;

  // Memory allocations
  double mass_ = 0;

public:
  RobotHandler();
  RobotHandler(const RobotHandlerSettings &settings);
  void initialize(const RobotHandlerSettings &settings);
  bool initialized_ = false;

  void updateInternalData(const Eigen::VectorXd &x);
  const Eigen::VectorXd &shapeState(const Eigen::VectorXd &q,
                                    const Eigen::VectorXd &v);
  void set_q0(const Eigen::VectorXd &q0);

  const pinocchio::FrameIndex &get_root_id() { return root_ids_; }
  const std::vector<pinocchio::FrameIndex> &get_ee_ids() {
    return end_effector_ids_;
  }
  const pinocchio::FrameIndex &get_ee_id(const unsigned long &i) {
    return end_effector_ids_[i];
  }

  const pinocchio::SE3 &get_ee_frame(const unsigned long &i) {
    return rdata_.oMf[get_ee_id(i)];
  };

  const pinocchio::SE3 &get_root_frame();

  void compute_mass();

  const double &get_mass() { return mass_; }
  const pinocchio::Model &get_rmodel() { return rmodel_; }
  const pinocchio::Model &get_rmodel_complete() { return rmodel_complete_; }
  const pinocchio::Data &get_rdata() { return rdata_; }
  const Eigen::VectorXd &get_q0() { return q0_; }
  const Eigen::VectorXd &get_v0() { return v0_; }
  const Eigen::VectorXd &get_q0Complete() { return q0Complete_; }
  const Eigen::VectorXd &get_v0Complete() { return v0Complete_; }
  const Eigen::VectorXd &get_x0() { return x0_; }

  const std::string &get_ee_name(const unsigned long &i) {
    return settings_.end_effector_names[i];
  }
  const std::vector<std::string> &get_ee_names() {
    return settings_.end_effector_names;
  }
  const RobotHandlerSettings &get_settings() { return settings_; }
  const std::vector<unsigned long> &get_controlledJointsIDs() {
    return controlled_joints_id_;
  }

  const pinocchio::SE3 &get_ee_pose(const unsigned long &i) {
    return rdata_.oMf[get_ee_id(i)];
  }
  const Eigen::Vector3d &get_com_position() { return com_position_; }
};

} // namespace simple_mpc
#endif // SIMPLE_MPC_ROBOT_HANDLER
