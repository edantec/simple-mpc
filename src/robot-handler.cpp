#include "simple-mpc/robot-handler.hpp"

#include <iostream>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
namespace simple_mpc {

RobotDataHandler::RobotDataHandler() {}

RobotDataHandler::RobotDataHandler(const RobotModelHandler &settings) {
  initialize(settings);
}

// void RobotDataHandler::initialize(const RobotModelHandler &settings) {
//   dmodel_handler = settings;

//   // COMPLETE MODEL //
//   if (dmodel_handler.robot_description.size() > 0) {
//     pinocchio::urdf::buildModelFromXML(dmodel_handler.robot_description,
//                                        JointModelFreeFlyer(), rmodel_complete_);
//     std::cout << "### Build pinocchio model from rosparam robot_description."
//               << std::endl;
//   } else if (dmodel_handler.urdf_path.size() > 0) {
//     pinocchio::urdf::buildModel(dmodel_handler.urdf_path, JointModelFreeFlyer(),
//                                 rmodel_complete_);
//     std::cout << "### Build pinocchio model from urdf file." << std::endl;
//   } else {
//     throw std::invalid_argument(
//         "the urdf file, or robotDescription must be specified.");
//   }
//   if (dmodel_handler.srdf_path.size() > 0) {
//     srdf::loadReferenceConfigurations(rmodel_complete_, dmodel_handler.srdf_path,
//                                       false);
//     if (settings.load_rotor) {
//       srdf::loadRotorParameters(rmodel_complete_, dmodel_handler.srdf_path, false);
//     }
//     q_complete_ =
//         rmodel_complete_.referenceConfigurations[settings.base_configuration];
//   } else {
//     q_complete_ = Eigen::VectorXd::Zero(rmodel_complete_.nq);
//   }
//   v_complete_ = Eigen::VectorXd::Zero(rmodel_complete_.nv);

//   // REDUCED MODEL //

//   // Check if listed joints belong to model
//   for (std::vector<std::string>::const_iterator it =
//            dmodel_handler.controlled_joints_names.begin();
//        it != dmodel_handler.controlled_joints_names.end(); ++it) {
//     const std::string &joint_name = *it;
//     std::cout << joint_name << std::endl;
//     std::cout << rmodel_complete_.getJointId(joint_name) << std::endl;
//     if (not(rmodel_complete_.existJointName(joint_name))) {
//       std::cout << "joint: " << joint_name << " does not belong to the model"
//                 << std::endl;
//     }
//   }

//   // making list of blocked joints
//   std::vector<unsigned long> locked_joints_id;
//   for (std::vector<std::string>::const_iterator it =
//            rmodel_complete_.names.begin() + 1;
//        it != rmodel_complete_.names.end(); ++it) {
//     const std::string &joint_name = *it;
//     if (std::find(dmodel_handler.controlled_joints_names.begin(),
//                   dmodel_handler.controlled_joints_names.end(),
//                   joint_name) == dmodel_handler.controlled_joints_names.end()) {
//       locked_joints_id.push_back(rmodel_complete_.getJointId(joint_name));
//     }
//   }

//   rmodel_ = buildReducedModel(rmodel_complete_, locked_joints_id, q_complete_);
//   root_ids_ = rmodel_.getFrameId(dmodel_handler.root_name);
//   for (std::size_t i = 0; i < dmodel_handler.end_effector_names.size(); i++) {
//     std::string name = dmodel_handler.end_effector_names[i];
//     end_effector_map_.insert({name, rmodel_.getFrameId(name)});
//     end_effector_ids_.push_back(rmodel_.getFrameId(name));
//     ref_end_effector_map_.insert(
//         {name, addFrameToBase(dmodel_handler.feet_to_base_trans[i], name + "_ref")});
//   }
//   rdata_ = Data(rmodel_);

//   if (dmodel_handler.srdf_path.size() > 0) {
//     srdf::loadReferenceConfigurations(rmodel_, dmodel_handler.srdf_path, false);
//     if (settings.load_rotor) {
//       srdf::loadRotorParameters(rmodel_, dmodel_handler.srdf_path, false);
//     }
//     q_ = rmodel_.referenceConfigurations[dmodel_handler.base_configuration];
//   } else {
//     q_ = Eigen::VectorXd::Zero(rmodel_.nq);
//   }
//   v_ = Eigen::VectorXd::Zero(rmodel_.nv);
//   x_.resize(rmodel_.nq + rmodel_.nv);
//   x_centroidal_.resize(9);

//   // Generating list of indices for controlled joints //
//   for (std::vector<std::string>::const_iterator it = rmodel_.names.begin() + 1;
//        it != rmodel_.names.end(); ++it) {
//     const std::string &joint_name = *it;
//     if (std::find(dmodel_handler.controlled_joints_names.begin(),
//                   dmodel_handler.controlled_joints_names.end(),
//                   joint_name) != dmodel_handler.controlled_joints_names.end()) {
//       controlled_joints_ids_.push_back(rmodel_complete_.getJointId(joint_name));
//     }
//   }
//   updateConfiguration(q_, true);
//   computeMass();
//   initialized_ = true;
// }

pinocchio::FrameIndex RobotModelHandler::addFrameToBase(Eigen::Vector3d translation, std::string name) {
  auto placement = pinocchio::SE3::Identity();
  placement.translation() = translation;

  auto new_frame = pinocchio::Frame(name, model.frames[root_id].parentJoint, root_id, placement, pinocchio::OP_FRAME);
  auto frame_id = model.addFrame(new_frame);

  return frame_id;
}

void RobotDataHandler::updateConfiguration(const Eigen::VectorXd &q,
                                       const bool updateJacobians) {
  if (q.size() != dmodel_handler.model.nq) {
    throw std::runtime_error(
        "q must have the dimensions of the robot configuration.");
  }
  q_ = q;
  x_ << q_, v_;
  updateInternalData(updateJacobians);
}

void RobotDataHandler::updateState(const Eigen::VectorXd &q,
                               const Eigen::VectorXd &v,
                               const bool updateJacobians) {
  if (q.size() != dmodel_handler.model.nq) {
    throw std::runtime_error(
        "q must have the dimensions of the robot configuration.");
  }
  if (v.size() != dmodel_handler.model.nv) {
    throw std::runtime_error(
        "v must have the dimensions of the robot velocity.");
  }
  q_ = q;
  v_ = v;
  x_ << q, v;
  updateInternalData(updateJacobians);
}

void RobotDataHandler::updateInternalData(const bool updateJacobians) {
  forwardKinematics(dmodel_handler.model, data, q_);
  updateFramePlacements(dmodel_handler.model, data);
  com_position_ = centerOfMass(dmodel_handler.model, data, q_, false);
  computeCentroidalMomentum(dmodel_handler.model, data, q_, v_);

  x_centroidal_.head(3) = com_position_;
  x_centroidal_.segment(3, 3) = data.hg.linear();
  x_centroidal_.tail(3) = data.hg.angular();

  if (updateJacobians)
    updateJacobiansMassMatrix();
}

void RobotDataHandler::updateJacobiansMassMatrix() {
  computeJointJacobians(dmodel_handler.model, data);
  computeJointJacobiansTimeVariation(dmodel_handler.model, data, q_, v_);
  crba(dmodel_handler.model, data, q_);
  make_symmetric(data.M);
  nonLinearEffects(dmodel_handler.model, data, q_, v_);
  dccrba(dmodel_handler.model, data, q_, v_);
}

const Eigen::VectorXd RobotDataHandler::shapeState(const Eigen::VectorXd &q,
                                               const Eigen::VectorXd &v) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dmodel_handler.model.nq + dmodel_handler.model.nv);
  if (q.size() == dmodel_handler.model.nq && v.size() == dmodel_handler.model.nv) {
    x.head<7>() = q.head<7>();
    x.segment<6>(dmodel_handler.model.nq) = v.head<6>();

    int i = 0;
    for (unsigned long jointID : dmodel_handler.controlled_joints_ids)
      if (jointID > 1) {
        x(i + 7) = q((long)jointID + 5);
        x(dmodel_handler.model.nq + i + 6) = v((long)jointID + 4);
        i++;
      }
    return x;
  } else if (q.size() == dmodel_handler.model.nq && v.size() == dmodel_handler.model.nv) {
    x << q, v;
    return x;
  } else {
    throw std::runtime_error(
        "q and v must have the dimentions of the reduced or complete model.");
  }
}

Eigen::VectorXd RobotDataHandler::difference(const Eigen::VectorXd &x1,
                                         const Eigen::VectorXd &x2) {
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(2 * dmodel_handler.model.nv);
  pinocchio::difference(dmodel_handler.model, x1.head(dmodel_handler.model.nq), x2.head(dmodel_handler.model.nq),
                        dx.head(dmodel_handler.model.nv));
  dx.tail(dmodel_handler.model.nq) = x2.tail(dmodel_handler.model.nq) - x1.tail(dmodel_handler.model.nq);

  return dx;
}

} // namespace simple_mpc
