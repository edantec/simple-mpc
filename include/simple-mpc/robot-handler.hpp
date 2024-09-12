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
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <vector>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {

/**
 * @brief Class managing every robot-related quantities
 */

struct RobotHandlerSettings {
public:
  std::string urdf_path = "";
  std::string srdf_path = "";
  std::string robot_description = "";
  std::vector<std::string> controlled_joints_names;

  std::vector<std::string> end_effector_names;
  std::string root_name = "";
  std::string base_configuration = "";
  bool load_rotor = false;
};

class RobotHandler {
private:
  RobotHandlerSettings settings_;

  // Vectors of usefull index
  std::vector<unsigned long> controlled_joints_ids_;
  std::map<std::string, pinocchio::FrameIndex> end_effector_map_;
  std::vector<pinocchio::FrameIndex> end_effector_ids_;
  unsigned long root_ids_;

  // Pinocchio objects
  pinocchio::Model rmodel_complete_, rmodel_;
  pinocchio::Data rdata_;

  // State vectors
  Eigen::VectorXd q0_complete_, q0_;
  Eigen::VectorXd v0_complete_, v0_;
  Eigen::VectorXd x0_;
  Eigen::VectorXd x_internal_;

  // Robot total mass and CoM
  double mass_ = 0;
  Eigen::Vector3d com_position_;

public:
  RobotHandler();
  RobotHandler(const RobotHandlerSettings &settings);
  void initialize(const RobotHandlerSettings &settings);
  bool initialized_ = false;

  // Set new robot state
  void updateInternalData(const Eigen::VectorXd &x);
  void setConfiguration(const Eigen::VectorXd &q0);

  // Return reduced state from measures
  const Eigen::VectorXd &shapeState(const Eigen::VectorXd &q,
                                    const Eigen::VectorXd &v);

  // Getters
  const pinocchio::FrameIndex &getRootId() { return root_ids_; }
  const std::vector<pinocchio::FrameIndex> &getFeetIds() {
    return end_effector_ids_;
  }
  const pinocchio::FrameIndex &getFootId(const std::string &ee_name) {
    return end_effector_map_.at(ee_name);
  }

  const pinocchio::SE3 &getFootPose(const std::string &ee_name) {
    return rdata_.oMf[getFootId(ee_name)];
  };

  const pinocchio::SE3 &getRootFrame();

  const double &getMass() { return mass_; }
  const pinocchio::Model &getModel() { return rmodel_; }
  const pinocchio::Model &getModelComplete() { return rmodel_complete_; }
  const pinocchio::Data &getData() { return rdata_; }
  const Eigen::VectorXd &getConfiguration() { return q0_; }
  const Eigen::VectorXd &getVelocity() { return v0_; }
  const Eigen::VectorXd &getCompleteConfiguration() { return q0_complete_; }
  const Eigen::VectorXd &getCompleteVelocity() { return v0_complete_; }
  const Eigen::VectorXd &getState() { return x0_; }

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

  // Compute the total robot mass
  void computeMass();
};

} // namespace simple_mpc
