#include "simple-mpc/robot-handler.hpp"

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
                                       pinocchio::JointModelFreeFlyer(),
                                       rmodel_complete_);
    std::cout << "### Build pinocchio model from rosparam robot_description."
              << std::endl;
  } else if (settings_.urdf_path.size() > 0) {
    pinocchio::urdf::buildModel(settings_.urdf_path,
                                pinocchio::JointModelFreeFlyer(),
                                rmodel_complete_);
    std::cout << "### Build pinocchio model from urdf file." << std::endl;
  } else {
    throw std::invalid_argument(
        "the urdf file, or robotDescription must be specified.");
  }

  pinocchio::srdf::loadReferenceConfigurations(rmodel_complete_,
                                               settings_.srdf_path, false);
  if (settings.load_rotor) {
    pinocchio::srdf::loadRotorParameters(rmodel_complete_, settings_.srdf_path,
                                         false);
  }
  q0_complete_ =
      rmodel_complete_.referenceConfigurations[settings.base_configuration];
  v0_complete_ = Eigen::VectorXd::Zero(rmodel_complete_.nv);

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

  rmodel_ = pinocchio::buildReducedModel(rmodel_complete_, locked_joints_id,
                                         q0_complete_);
  for (auto &name : settings_.end_effector_names) {
    end_effector_map_.insert({name, rmodel_.getFrameId(name)});
    end_effector_ids_.push_back(rmodel_.getFrameId(name));
  }
  root_ids_ = rmodel_.getFrameId(settings_.root_name);
  rdata_ = pinocchio::Data(rmodel_);

  pinocchio::srdf::loadReferenceConfigurations(rmodel_, settings_.srdf_path,
                                               false);
  if (settings.load_rotor) {
    pinocchio::srdf::loadRotorParameters(rmodel_, settings_.srdf_path, false);
  }
  q0_ = rmodel_.referenceConfigurations[settings_.base_configuration];
  v0_ = Eigen::VectorXd::Zero(rmodel_.nv);
  x0_.resize(rmodel_.nq + rmodel_.nv);
  x_internal_.resize(rmodel_.nq + rmodel_.nv);
  x0_ << q0_, v0_;
  x_internal_ << q0_, v0_;
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

  updateInternalData(q0_);
  computeMass();
  initialized_ = true;
}

void RobotHandler::setConfiguration(const Eigen::VectorXd &q0) {
  q0_ = q0;
  x0_ << q0_, v0_;
  updateInternalData(q0_);
}

void RobotHandler::updateInternalData(const Eigen::VectorXd &x) {
  /** x is the reduced posture, or contains the reduced posture in the first
   * elements */
  pinocchio::forwardKinematics(rmodel_, rdata_, x.head(rmodel_.nq));
  pinocchio::updateFramePlacements(rmodel_, rdata_);
  com_position_ =
      pinocchio::centerOfMass(rmodel_, rdata_, x.head(rmodel_.nq), false);
  pinocchio::computeCentroidalMomentum(rmodel_, rdata_, x.head(rmodel_.nq),
                                       x.tail(rmodel_.nv));
}

const Eigen::VectorXd &RobotHandler::shapeState(const Eigen::VectorXd &q,
                                                const Eigen::VectorXd &v) {
  if (q.size() == rmodel_complete_.nq && v.size() == rmodel_complete_.nv) {
    x_internal_.head<7>() = q.head<7>();
    x_internal_.segment<6>(rmodel_.nq) = v.head<6>();

    int i = 0;
    for (unsigned long jointID : controlled_joints_ids_)
      if (jointID > 1) {
        x_internal_(i + 7) = q((long)jointID + 5);
        x_internal_(rmodel_.nq + i + 6) = v((long)jointID + 4);
        i++;
      }
    return x_internal_;
  } else if (q.size() == rmodel_.nq && v.size() == rmodel_.nv) {
    x_internal_ << q, v;
    return x_internal_;
  } else {
    throw std::runtime_error(
        "q and v must have the dimentions of the reduced or complete model.");
  }
}

void RobotHandler::computeMass() {
  mass_ = 0;
  for (pinocchio::Inertia &I : rmodel_.inertias)
    mass_ += I.mass();
}

} // namespace simple_mpc
