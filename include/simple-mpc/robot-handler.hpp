///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <pinocchio/fwd.hpp>
// Include pinocchio first
#include <Eigen/Dense>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <vector>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace pinocchio;
/**
 * @brief Class managing every robot-related quantities.
 *
 * It holds the robot data, controlled joints, end-effector names
 * and other useful items.
 */

struct RobotHandlerSettings {
public:
  // Path to fetch the robot model.
  // Either urdf_path or robot_description should
  // be filled.
  std::string urdf_path = "";
  std::string srdf_path = "";
  std::string robot_description = "";

  // Joint-related items
  std::vector<std::string> controlled_joints_names;
  std::vector<std::string> end_effector_names;

  // Useful names
  std::string root_name = "";
  std::string base_configuration = "";

  // Wether to use rotor parameters in joint dynamics
  bool load_rotor = false;
};

class RobotHandler {
private:
  RobotHandlerSettings settings_;

  // Useful index
  std::vector<unsigned long> controlled_joints_ids_;
  std::map<std::string, FrameIndex> end_effector_map_;
  std::vector<FrameIndex> end_effector_ids_;
  unsigned long root_ids_;

  // Pinocchio objects
  Model rmodel_complete_, rmodel_;
  Data rdata_;

  // State vectors
  Eigen::VectorXd q_complete_, q_;
  Eigen::VectorXd v_complete_, v_;
  Eigen::VectorXd x_;
  Eigen::VectorXd x_centroidal_;
  Eigen::MatrixXd M_; // Mass matrix

  // Robot total mass and CoM
  double mass_ = 0;
  Eigen::Vector3d com_position_;

public:
  RobotHandler();
  RobotHandler(const RobotHandlerSettings &settings);
  void initialize(const RobotHandlerSettings &settings);
  bool initialized_ = false;

  // Set new robot state
  void updateConfiguration(const Eigen::VectorXd &q,
                           const bool updateJacobians);
  void updateState(const Eigen::VectorXd &x, const bool updateJacobians);
  void updateInternalData(const bool updateJacobians);
  void updateJacobiansMassMatrix();

  // Return reduced state from measures
  const Eigen::VectorXd shapeState(const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &v);

  // Getters
  const FrameIndex &getRootId() { return root_ids_; }
  const std::vector<FrameIndex> &getFeetIds() { return end_effector_ids_; }
  const FrameIndex &getFootId(const std::string &ee_name) {
    return end_effector_map_.at(ee_name);
  }
  const SE3 &getFootPose(const std::string &ee_name) {
    return rdata_.oMf[getFootId(ee_name)];
  };
  const SE3 &getRootFrame() { return rdata_.oMf[root_ids_]; }
  const Eigen::VectorXd &getCentroidalState() { return x_centroidal_; }
  const double &getMass() { return mass_; }
  const Model &getModel() { return rmodel_; }
  const Model &getModelComplete() { return rmodel_complete_; }
  const Data &getData() { return rdata_; }
  const Eigen::VectorXd &getConfiguration() { return q_; }
  const Eigen::VectorXd &getVelocity() { return v_; }
  const Eigen::VectorXd &getCompleteConfiguration() { return q_complete_; }
  const Eigen::VectorXd &getCompleteVelocity() { return v_complete_; }
  const Eigen::VectorXd &getState() { return x_; }
  const std::string &getFootName(const unsigned long &i) {
    return settings_.end_effector_names[i];
  }
  const std::vector<std::string> &getFeetNames() {
    return settings_.end_effector_names;
  }
  const RobotHandlerSettings &getSettings() { return settings_; }
  const std::vector<unsigned long> &getControlledJointsIDs() {
    return controlled_joints_ids_;
  }
  const Eigen::Vector3d &getComPosition() { return com_position_; }
  const Eigen::MatrixXd &getMassMatrix() { return M_; }

  // Compute the total robot mass
  void computeMass();
};

} // namespace simple_mpc
