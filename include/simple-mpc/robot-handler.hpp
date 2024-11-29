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
  /**
   * @brief Robot model with all joints unlocked
   */
  Model model_full;

  /**
   * @brief Reduced model to be used by ocp
   */
  Model model;

  /**
   * @brief Robot total mass
   */
  double mass;

  /**
   * @brief Joint id to be controlled in full model
   */
  std::vector<unsigned long> controlled_joints_ids;

  /**
   * @brief Ids of the frames to be in contact with the environment
   */
  std::vector<FrameIndex> feet_ids;

  /**
   * @brief Ids of the frames that are reference position for the feet
   */
  std::vector<FrameIndex> ref_feet_ids;

  /**
   * @brief Name of the configuration to use as reference
   */
  Eigen::VectorXd reference_configuration;

  /**
   * @brief Root frame id
   */
  pinocchio::FrameIndex root_id;
};

class RobotHandler {
private:
  RobotHandlerSettings settings_;
  Data data;

  // State vectors
  Eigen::VectorXd q_complete_, q_;
  Eigen::VectorXd v_complete_, v_;
  Eigen::VectorXd x_;
  Eigen::VectorXd x_centroidal_;
  Eigen::MatrixXd M_; // Mass matrix

  // Position of robot Center of Mass
  Eigen::Vector3d com_position_;

public:
  RobotHandler();
  RobotHandler(const RobotHandlerSettings &settings);
  void initialize(const RobotHandlerSettings &settings);
  bool initialized_ = false;

  // Set new robot state
  void updateConfiguration(const Eigen::VectorXd &q,
                           const bool updateJacobians);
  void updateState(const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                   const bool updateJacobians);
  void updateInternalData(const bool updateJacobians);
  void updateJacobiansMassMatrix();

  pinocchio::FrameIndex addFrameToBase(Eigen::Vector3d translation,
                                       std::string name);

  // Return reduced state from measures
  const Eigen::VectorXd shapeState(const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &v);

  Eigen::VectorXd difference(const Eigen::VectorXd &x1,
                             const Eigen::VectorXd &x2);
  // Getters
  const std::vector<FrameIndex> &getFeetIds() { return settings_.ref_feet_ids; }
  const FrameIndex &getFootId(size_t feet) {
    return settings_.feet_ids.at(feet);
  }
  const SE3 &getFootPose(size_t feet) {
    return data.oMf[getFootId(feet)];
  };
  const FrameIndex &getRefFootId(size_t feet) {
    return settings_.feet_ids.at(feet);
  }
  const SE3 &getRefFootPose(size_t feet) {
    return data.oMf[getRefFootId(feet)];
  };
  const SE3 &getRootFrame() { return data.oMf[settings_.root_id]; }
  const Eigen::VectorXd &getCentroidalState() { return x_centroidal_; }
  const Model &getModel() { return settings_.model; }
  const Model &getCompleteModel() { return settings_.model_full; }
  const Data &getData() { return data; }
  const Eigen::VectorXd &getConfiguration() { return q_; }
  const Eigen::VectorXd &getVelocity() { return v_; }
  const Eigen::VectorXd &getCompleteConfiguration() { return q_complete_; }
  const Eigen::VectorXd &getCompleteVelocity() { return v_complete_; }
  const Eigen::VectorXd &getState() { return x_; }
  const RobotHandlerSettings &getSettings() { return settings_; }
  const Eigen::Vector3d &getComPosition() { return com_position_; }
  const Eigen::MatrixXd &getMassMatrix() { return data.M; }
  // Compute the total robot mass
  void computeMass();
};

} // namespace simple_mpc
