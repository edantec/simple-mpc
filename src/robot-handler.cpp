#include "simple-mpc/robot-handler.hpp"

#include <iostream>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
namespace simple_mpc {

RobotHandler::RobotHandler() {}

RobotHandler::RobotHandler(const RobotHandlerSettings &settings) {
  initialize(settings);
}

void RobotHandler::initialize(const RobotHandlerSettings &settings) {
  settings_ = settings;

  // COMPLETE MODEL //
  if (settings_.robot_description.size() > 0) {
    pinocchio::urdf::buildModelFromXML(settings_.robot_description,
                                       JointModelFreeFlyer(), rmodel_complete_);
    std::cout << "### Build pinocchio model from rosparam robot_description."
              << std::endl;
  } else if (settings_.urdf_path.size() > 0) {
    pinocchio::urdf::buildModel(settings_.urdf_path, JointModelFreeFlyer(),
                                rmodel_complete_);
    std::cout << "### Build pinocchio model from urdf file." << std::endl;
  } else {
    throw std::invalid_argument(
        "the urdf file, or robotDescription must be specified.");
  }
  if (settings_.srdf_path.size() > 0) {
    srdf::loadReferenceConfigurations(rmodel_complete_, settings_.srdf_path,
                                      false);
    if (settings.load_rotor) {
      srdf::loadRotorParameters(rmodel_complete_, settings_.srdf_path, false);
    }
    q_complete_ =
        rmodel_complete_.referenceConfigurations[settings.base_configuration];
  } else {
    q_complete_ = Eigen::VectorXd::Zero(rmodel_complete_.nq);
  }
  v_complete_ = Eigen::VectorXd::Zero(rmodel_complete_.nv);

  // REDUCED MODEL //

  if (settings_.controlled_joints_names[0] != settings.root_name) {
    throw std::invalid_argument("the joint at index 0 must be called " +
                                settings.root_name);
  }

  // Check if listed joints belong to model
  for (std::vector<std::string>::const_iterator it =
           settings_.controlled_joints_names.begin();
       it != settings_.controlled_joints_names.end(); ++it) {
    const std::string &joint_name = *it;
    std::cout << joint_name << std::endl;
    std::cout << rmodel_complete_.getJointId(joint_name) << std::endl;
    if (not(rmodel_complete_.existJointName(joint_name))) {
      std::cout << "joint: " << joint_name << " does not belong to the model"
                << std::endl;
    }
  }

  // making list of blocked joints
  std::vector<unsigned long> locked_joints_id;
  for (std::vector<std::string>::const_iterator it =
           rmodel_complete_.names.begin() + 1;
       it != rmodel_complete_.names.end(); ++it) {
    const std::string &joint_name = *it;
    if (std::find(settings_.controlled_joints_names.begin(),
                  settings_.controlled_joints_names.end(),
                  joint_name) == settings_.controlled_joints_names.end()) {
      locked_joints_id.push_back(rmodel_complete_.getJointId(joint_name));
    }
  }

  rmodel_ = buildReducedModel(rmodel_complete_, locked_joints_id, q_complete_);
  for (auto &name : settings_.end_effector_names) {
    end_effector_map_.insert({name, rmodel_.getFrameId(name)});
    end_effector_ids_.push_back(rmodel_.getFrameId(name));
  }
  root_ids_ = rmodel_.getFrameId(settings_.root_name);
  rdata_ = Data(rmodel_);

  if (settings_.srdf_path.size() > 0) {
    srdf::loadReferenceConfigurations(rmodel_, settings_.srdf_path, false);
    if (settings.load_rotor) {
      srdf::loadRotorParameters(rmodel_, settings_.srdf_path, false);
    }
    q_ = rmodel_.referenceConfigurations[settings_.base_configuration];
  } else {
    q_ = settings_.vector_configuration;
  }
  v_ = Eigen::VectorXd::Zero(rmodel_.nv);
  x_.resize(rmodel_.nq + rmodel_.nv);
  x_centroidal_.resize(9);
  // Generating list of indices for controlled joints //
  for (std::vector<std::string>::const_iterator it = rmodel_.names.begin() + 1;
       it != rmodel_.names.end(); ++it) {
    const std::string &joint_name = *it;
    if (std::find(settings_.controlled_joints_names.begin(),
                  settings_.controlled_joints_names.end(),
                  joint_name) != settings_.controlled_joints_names.end()) {
      controlled_joints_ids_.push_back(rmodel_complete_.getJointId(joint_name));
    }
  }
  M_.resize(rmodel_.nv, rmodel_.nv);
  updateConfiguration(q_, true);
  computeMass();
  initialized_ = true;
}

void RobotHandler::updateConfiguration(const Eigen::VectorXd &q,
                                       const bool updateJacobians) {
  if (q.size() != rmodel_.nq) {
    throw std::runtime_error(
        "q must have the dimensions of the robot configuration.");
  }
  q_ = q;
  x_ << q_, v_;
  updateInternalData(updateJacobians);
}

void RobotHandler::updateState(const Eigen::VectorXd &q,
                               const Eigen::VectorXd &v,
                               const bool updateJacobians) {
  if (q.size() != rmodel_.nq) {
    throw std::runtime_error(
        "q must have the dimensions of the robot configuration.");
  }
  if (v.size() != rmodel_.nv) {
    throw std::runtime_error(
        "v must have the dimensions of the robot velocity.");
  }
  q_ = q;
  v_ = v;
  x_ << q, v;
  updateInternalData(updateJacobians);
}

void RobotHandler::updateInternalData(const bool updateJacobians) {
  forwardKinematics(rmodel_, rdata_, q_);
  updateFramePlacements(rmodel_, rdata_);
  com_position_ = centerOfMass(rmodel_, rdata_, q_, false);
  computeCentroidalMomentum(rmodel_, rdata_, q_, v_);

  x_centroidal_.head(3) = com_position_;
  x_centroidal_.segment(3, 3) = rdata_.hg.linear();
  x_centroidal_.tail(3) = rdata_.hg.angular();

  if (updateJacobians)
    updateJacobiansMassMatrix();
}

void RobotHandler::updateJacobiansMassMatrix() {
  computeJointJacobians(rmodel_, rdata_);
  computeJointJacobiansTimeVariation(rmodel_, rdata_, q_, v_);
  crba(rmodel_, rdata_, q_);
  make_symmetric(rdata_.M);
  nonLinearEffects(rmodel_, rdata_, q_, v_);
  dccrba(rmodel_, rdata_, q_, v_);
}

const Eigen::VectorXd RobotHandler::shapeState(const Eigen::VectorXd &q,
                                               const Eigen::VectorXd &v) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(rmodel_.nq + rmodel_.nv);
  if (q.size() == rmodel_complete_.nq && v.size() == rmodel_complete_.nv) {
    x.head<7>() = q.head<7>();
    x.segment<6>(rmodel_.nq) = v.head<6>();

    int i = 0;
    for (unsigned long jointID : controlled_joints_ids_)
      if (jointID > 1) {
        x(i + 7) = q((long)jointID + 5);
        x(rmodel_.nq + i + 6) = v((long)jointID + 4);
        i++;
      }
    return x;
  } else if (q.size() == rmodel_.nq && v.size() == rmodel_.nv) {
    x << q, v;
    return x;
  } else {
    throw std::runtime_error(
        "q and v must have the dimentions of the reduced or complete model.");
  }
}

void RobotHandler::computeMass() {
  mass_ = 0;
  for (Inertia &I : rmodel_.inertias)
    mass_ += I.mass();
}

Eigen::VectorXd RobotHandler::difference(const Eigen::VectorXd &x1,
                                         const Eigen::VectorXd &x2) {
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(2 * rmodel_.nv);
  pinocchio::difference(rmodel_, x1.head(rmodel_.nq), x2.head(rmodel_.nq),
                        dx.head(rmodel_.nv));
  dx.tail(rmodel_.nq) = x2.tail(rmodel_.nq) - x1.tail(rmodel_.nq);

  return dx;
}

} // namespace simple_mpc
