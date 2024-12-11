#include "simple-mpc/robot-handler.hpp"

#include <iostream>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
namespace simple_mpc {

  RobotModelHandler::RobotModelHandler(const Model& model, const std::string& reference_configuration_name, const std::string& base_frame_name, const std::vector<std::string>& locked_joint_names)
  : model_full_(model)
  {
    // Construct controlled and locked joints ids list
    std::vector<unsigned long> locked_joint_ids;
    for(size_t i = 1; i< model_full_.names.size(); i++)
    {
      const std::string joint_name = model_full_.names.at(i);
      if(count(locked_joint_names.begin(), locked_joint_names.end(), joint_name) == 0)
      {
        controlled_joints_ids_.push_back(model_full_.getJointId(joint_name));
      }
      else
      {
        locked_joint_ids.push_back(model_full_.getJointId(joint_name));
      }
    }

    // Build reduced model with locked joints
    buildReducedModel(model_full_, locked_joint_ids, model_full_.referenceConfigurations[reference_configuration_name], model_);

    // Root frame id
    base_id_ = model_.getFrameId(base_frame_name);

    // Set reference state
    reference_state_ = shapeState(model_full_.referenceConfigurations[reference_configuration_name], Eigen::VectorXd::Zero(model_full_.nv));

    // Mass
    mass_ = pinocchio::computeTotalMass(model_);
  }

FrameIndex RobotModelHandler::addFoot(const std::string& foot_name, const std::string& placement_reference_frame_name, const SE3& placement)
{
  feet_names_.push_back(foot_name);
  feet_ids_.push_back(model_.getFrameId(foot_name));

  // Create reference frame
  FrameIndex placement_reference_frame_id = model_.getFrameId(placement_reference_frame_name);
  JointIndex parent_joint = model_.frames[placement_reference_frame_id].parentJoint;

  auto new_frame = pinocchio::Frame(foot_name + "_ref", parent_joint, placement_reference_frame_id, placement, pinocchio::OP_FRAME);
  auto frame_id = model_.addFrame(new_frame);

  ref_feet_ids_.push_back(frame_id);

  return frame_id;
}


Eigen::VectorXd RobotModelHandler::shapeState(const Eigen::VectorXd &q, const Eigen::VectorXd &v) const
{
  const size_t nq_full = model_full_.nq;
  const size_t nv_full = model_full_.nv;
  const size_t nq = model_.nq;
  const size_t nv = model_.nv;
  const size_t nx = nq + nv;
  Eigen::VectorXd x(nx);

  assert(nq_full == q.size() && "Configuration vector has wrong size.");
  assert(nv_full == v.size() && "Velocity vector has wrong size.");

  // Copy each controlled joint to state vector
  int iq = 0;
  int iv = nq;
  for (unsigned long jointId : controlled_joints_ids_)
  {
    const size_t j_idx_q = model_full_.idx_qs[jointId];
    const size_t j_idx_v = model_full_.idx_vs[jointId];
    const size_t j_nq = model_full_.nqs[jointId];
    const size_t j_nv = model_full_.nvs[jointId];

    x.segment(iq, j_nq) = q.segment(j_idx_q, j_nq);
    x.segment(iv, j_nv) = v.segment(j_idx_v, j_nv);

    iq += j_nq;
    iv += j_nv;
  }
  return x;
}

Eigen::VectorXd RobotModelHandler::difference(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{
  const size_t nq = model_.nq;
  const size_t nv = model_.nv;
  const size_t ndx = 2* nv;

  Eigen::VectorXd dx(ndx);

  // Difference over q
  pinocchio::difference(model_, x1.head(nq), x2.head(nq), dx.head(nv));

  // Difference over v
  dx.tail(nv) = x2.tail(nv) - x1.tail(nv);

  return dx;
}

RobotDataHandler::RobotDataHandler(const RobotModelHandler &model_handler)
: model_handler_(model_handler)
, data_(model_handler.getModel())
{
  updateInternalData(model_handler.getReferenceState(), true);
}

void RobotDataHandler::updateInternalData(const Eigen::VectorXd &x, const bool updateJacobians) {
  const Eigen::Block q = x.head(model_handler_.getModel().nq);
  const Eigen::Block v = x.tail(model_handler_.getModel().nv);

  forwardKinematics(model_handler_.getModel(), data_, q);
  updateFramePlacements(model_handler_.getModel(), data_);
  computeCentroidalMomentum(model_handler_.getModel(), data_, q, v);

  if (updateJacobians)
  {
    updateJacobiansMassMatrix(x);
  }
}

void RobotDataHandler::updateJacobiansMassMatrix(const Eigen::VectorXd &x) {
  const Eigen::Block q = x.head(model_handler_.getModel().nq);
  const Eigen::Block v = x.tail(model_handler_.getModel().nv);

  computeJointJacobians(model_handler_.getModel(), data_);
  computeJointJacobiansTimeVariation(model_handler_.getModel(), data_, q, v);
  crba(model_handler_.getModel(), data_, q);
  data_.M.triangularView<Eigen::StrictlyLower>() =
      data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  nonLinearEffects(model_handler_.getModel(), data_, q, v);
  dccrba(model_handler_.getModel(), data_, q, v);
}

RobotDataHandler::CentroidalStateVector RobotDataHandler::getCentroidalState() const
{
  RobotDataHandler::CentroidalStateVector x_centroidal;
  x_centroidal.head(3) = data_.com[0];
  x_centroidal.segment(3, 3) = data_.hg.linear();
  x_centroidal.tail(3) = data_.hg.angular();
  return x_centroidal;
}

} // namespace simple_mpc
