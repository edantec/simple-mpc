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
struct RobotModelHandler {
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
   * @brief Names of the frames to be in contact with the environment
   */
  std::vector<std::string> feet_names;

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

public:
  // Helper function to augment the model
  pinocchio::FrameIndex addFrameToBase(Eigen::Vector3d translation, std::string name);

  // Const getters
  size_t getFootIndex(const std::string &foot_name) const
  {
    std::find(feet_names.begin(), feet_names.end(), foot_name) - feet_names.begin();
  }

  const std::string &getFootName(size_t i) const
  {
    return feet_names.at(i);
  }

  const std::vector<std::string> &getFeetNames() const
  {
    return feet_names;
  }

  const FrameIndex &getFootId(const std::string &foot_name) const
  {
    return feet_ids.at(getFootIndex(foot_name));
  }

  const FrameIndex &getRefFootId(const std::string &foot_name) const
  {
    return feet_ids.at(getFootIndex(foot_name));
  }

  double getMass() const
  {
    return mass;
  }

  const Model &getModel()
  {
    return model;
  }

  const Model &getCompleteModel()
  {
    return model_full;
  }
};

class RobotDataHandler {
private:
  RobotModelHandler model_handler;
  Data data;

  // State vectors
  Eigen::VectorXd q_complete_, q_;
  Eigen::VectorXd v_complete_, v_;
  Eigen::VectorXd x_;
  Eigen::VectorXd x_centroidal_;

public:
  RobotDataHandler();
  RobotDataHandler(const RobotModelHandler &settings);
  void initialize(const RobotModelHandler &settings);
  bool initialized_ = false;

  // Set new robot state
  void updateConfiguration(const Eigen::VectorXd &q, const bool updateJacobians);
  void updateState(const Eigen::VectorXd &q, const Eigen::VectorXd &v, const bool updateJacobians);
  void updateInternalData(const bool updateJacobians);
  void updateJacobiansMassMatrix();

  // Return reduced state from measures
  const Eigen::VectorXd shapeState(const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &v);

  Eigen::VectorXd difference(const Eigen::VectorXd &x1,
                             const Eigen::VectorXd &x2);
  // Getters
  const SE3 &getRefFootPose(const std::string &foot_name) const
  {
    return data.oMf[model_handler.getRefFootId(foot_name)];
  };
  const SE3 &getFootPose(const std::string &foot_name) const
  {
    return data.oMf[model_handler.getFootId(foot_name)];
  };
  const SE3 &getRootFramePose() {
    return data.oMf[model_handler.root_id];
  }

  const Eigen::VectorXd &getConfiguration() { return q_; }
  const Eigen::VectorXd &getVelocity() { return v_; }
  const Eigen::VectorXd &getCompleteConfiguration() { return q_complete_; }
  const Eigen::VectorXd &getCompleteVelocity() { return v_complete_; }
  const Eigen::VectorXd &getCentroidalState() { return x_centroidal_; }
  const Eigen::VectorXd &getState() { return x_; }

  const Eigen::Vector3d &getComPosition() { return data.com[0]; }
  const Eigen::MatrixXd &getMassMatrix() { return data.M; }

  const RobotModelHandler &getModelHandler() { return model_handler; }
  const Data &getData() { return data; }
};

} // namespace simple_mpc
