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
//   model_handler = settings;

//   // COMPLETE MODEL //
//   if (model_handler.robot_description.size() > 0) {
//     pinocchio::urdf::buildModelFromXML(model_handler.robot_description,
//                                        JointModelFreeFlyer(), rmodel_complete_);
//     std::cout << "### Build pinocchio model from rosparam robot_description."
//               << std::endl;
//   } else if (model_handler.urdf_path.size() > 0) {
//     pinocchio::urdf::buildModel(model_handler.urdf_path, JointModelFreeFlyer(),
//                                 rmodel_complete_);
//     std::cout << "### Build pinocchio model from urdf file." << std::endl;
//   } else {
//     throw std::invalid_argument(
//         "the urdf file, or robotDescription must be specified.");
//   }
//   if (model_handler.srdf_path.size() > 0) {
//     srdf::loadReferenceConfigurations(rmodel_complete_, model_handler.srdf_path,
//                                       false);
//     if (settings.load_rotor) {
//       srdf::loadRotorParameters(rmodel_complete_, model_handler.srdf_path, false);
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
//            model_handler.controlled_joints_names.begin();
//        it != model_handler.controlled_joints_names.end(); ++it) {
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
//     if (std::find(model_handler.controlled_joints_names.begin(),
//                   model_handler.controlled_joints_names.end(),
//                   joint_name) == model_handler.controlled_joints_names.end()) {
//       locked_joints_id.push_back(rmodel_complete_.getJointId(joint_name));
//     }
//   }

//   rmodel_ = buildReducedModel(rmodel_complete_, locked_joints_id, q_complete_);
//   root_ids_ = rmodel_.getFrameId(model_handler.root_name);
//   for (std::size_t i = 0; i < model_handler.end_effector_names.size(); i++) {
//     std::string name = model_handler.end_effector_names[i];
//     end_effector_map_.insert({name, rmodel_.getFrameId(name)});
//     end_effector_ids_.push_back(rmodel_.getFrameId(name));
//     ref_end_effector_map_.insert(
//         {name, addFrameToBase(model_handler.feet_to_base_trans[i], name + "_ref")});
//   }
//   rdata_ = Data(rmodel_);

//   if (model_handler.srdf_path.size() > 0) {
//     srdf::loadReferenceConfigurations(rmodel_, model_handler.srdf_path, false);
//     if (settings.load_rotor) {
//       srdf::loadRotorParameters(rmodel_, model_handler.srdf_path, false);
//     }
//     q_ = rmodel_.referenceConfigurations[model_handler.base_configuration];
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
//     if (std::find(model_handler.controlled_joints_names.begin(),
//                   model_handler.controlled_joints_names.end(),
//                   joint_name) != model_handler.controlled_joints_names.end()) {
//       controlled_joints_ids_.push_back(rmodel_complete_.getJointId(joint_name));
//     }
//   }
//   updateConfiguration(q_, true);
//   computeMass();
//   initialized_ = true;
// }

const Eigen::VectorXd RobotModelHandler::shapeState(const Eigen::VectorXd &q, const Eigen::VectorXd &v) {
  const size_t nq_full = model_full.nq;
  const size_t nv_full = model_full.nv;
  const size_t nq = model.nq;
  const size_t nv = model.nv;
  const size_t nx = nq + nv;
  Eigen::VectorXd x(nx);

  assert(nq_full == q.size() && "Configuration vector has wrong size.");
  assert(nv_full == v.size() && "Velocity vector has wrong size.");

  // Floating base
  x.head<7>() = q.head<7>();
  x.segment<6>(nq) = v.head<6>();

  // Copy each controlled joint to state vector
  int iq = 7;
  int iv = nq + 6;
  for (unsigned long jointId : model_handler.controlled_joints_ids)
  {
    const size_t j_idx_q = model_full.idx_qs[jointId];
    const size_t j_idx_v = model_full.idx_vs[jointId];
    const size_t j_nq = model_full.nqs[jointId];
    const size_t j_nv = model_full.nvs[jointId];

    x.segment(iq, j_nq) = q.segment(j_idx_q, j_nq);
    x.segment(iv, j_nv) = v.segment(j_idx_v, j_nv);

    iq += j_nq;
    iv += j_nv;
  }
  return x;
}

Eigen::VectorXd RobotModelHandler::difference(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) {
  const size_t nq = model_handler.model.nq;
  const size_t nv = model_handler.model.nv;
  const size_t ndx = 2* nv;
  Eigen::VectorXd dx(nx);

  // Difference over q
  pinocchio::difference(model_handler.model, x1.head(nq), x2.head(nq), dx.head(model_handler.model.nv));

  // Difference over v
  dx.tail(nv) = x2.tail(nv) - x1.tail(nv);

  return dx;
}

pinocchio::FrameIndex RobotModelHandler::addFrameToBase(Eigen::Vector3d translation, std::string name) {
  auto placement = pinocchio::SE3::Identity();
  placement.translation() = translation;

  auto new_frame = pinocchio::Frame(name, model.frames[root_id].parentJoint, root_id, placement, pinocchio::OP_FRAME);
  auto frame_id = model.addFrame(new_frame);

  return frame_id;
}

void RobotDataHandler::updateConfiguration(const Eigen::VectorXd &q,
                                       const bool updateJacobians) {
  if (q.size() != model_handler.model.nq) {
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
  if (q.size() != model_handler.model.nq) {
    throw std::runtime_error(
        "q must have the dimensions of the robot configuration.");
  }
  if (v.size() != model_handler.model.nv) {
    throw std::runtime_error(
        "v must have the dimensions of the robot velocity.");
  }
  q_ = q;
  v_ = v;
  x_ << q, v;
  updateInternalData(updateJacobians);
}

void RobotDataHandler::updateInternalData(const bool updateJacobians) {
  forwardKinematics(model_handler.model, data, q_);
  updateFramePlacements(model_handler.model, data);
  com_position_ = centerOfMass(model_handler.model, data, q_, false);
  computeCentroidalMomentum(model_handler.model, data, q_, v_);

  x_centroidal_.head(3) = com_position_;
  x_centroidal_.segment(3, 3) = data.hg.linear();
  x_centroidal_.tail(3) = data.hg.angular();

  if (updateJacobians)
    updateJacobiansMassMatrix();
}

void RobotDataHandler::updateJacobiansMassMatrix() {
  computeJointJacobians(model_handler.model, data);
  computeJointJacobiansTimeVariation(model_handler.model, data, q_, v_);
  crba(model_handler.model, data, q_);
  make_symmetric(data.M);
  nonLinearEffects(model_handler.model, data, q_, v_);
  dccrba(model_handler.model, data, q_, v_);
}

} // namespace simple_mpc
