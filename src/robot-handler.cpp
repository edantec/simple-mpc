#include "simple-mpc/robot-handler.hpp"

#include <iostream>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
namespace simple_mpc
{

  RobotModelHandler::RobotModelHandler(
    const Model & model, const std::string & reference_configuration_name, const std::string & base_frame_name)
  : model_(model)
  {
    // Controlled joints index
    for (auto joint_name : model_.names)
      controlled_joints_ids_.push_back(model.getJointId(joint_name));

    // Root frame id
    base_id_ = model_.getFrameId(base_frame_name);

    // Set reference state
    reference_state_.resize(model_.nq + model_.nv);
    reference_state_ << model_.referenceConfigurations[reference_configuration_name], Eigen::VectorXd::Zero(model_.nv);

    // Mass
    mass_ = pinocchio::computeTotalMass(model_);
  }

  FrameIndex RobotModelHandler::addFoot(
    const std::string & foot_name, const std::string & placement_reference_frame_name, const SE3 & placement)
  {
    feet_names_.push_back(foot_name);
    feet_ids_.push_back(model_.getFrameId(foot_name));

    // Create reference frame
    FrameIndex placement_reference_frame_id = model_.getFrameId(placement_reference_frame_name);
    JointIndex parent_joint = model_.frames[placement_reference_frame_id].parentJoint;

    auto new_frame =
      pinocchio::Frame(foot_name + "_ref", parent_joint, placement_reference_frame_id, placement, pinocchio::OP_FRAME);
    auto frame_id = model_.addFrame(new_frame);

    ref_feet_ids_.push_back(frame_id);

    return frame_id;
  }

  Eigen::VectorXd RobotModelHandler::difference(const ConstVectorRef & x1, const ConstVectorRef & x2) const
  {
    const size_t nq = (size_t)model_.nq;
    const size_t nv = (size_t)model_.nv;
    const size_t ndx = 2 * nv;

    Eigen::VectorXd dx(ndx);

    // Difference over q
    pinocchio::difference(model_, x1.head(nq), x2.head(nq), dx.head(nv));

    // Difference over v
    dx.tail(nv) = x2.tail(nv) - x1.tail(nv);

    return dx;
  }

  std::vector<std::string> RobotModelHandler::getControlledJointNames() const
  {
    std::vector<std::string> joint_names;
    for (JointIndex id : controlled_joints_ids_)
    {
      joint_names.push_back(model_.names.at(id));
    }
    return joint_names;
  }

  RobotDataHandler::RobotDataHandler(const RobotModelHandler & model_handler)
  : model_handler_(model_handler)
  , data_(model_handler.getModel())
  {
    updateInternalData(model_handler.getReferenceState(), true);
  }

  void RobotDataHandler::updateInternalData(const ConstVectorRef & x, const bool updateJacobians)
  {
    const Eigen::Block q = x.head(model_handler_.getModel().nq);
    const Eigen::Block v = x.tail(model_handler_.getModel().nv);
    x_ = x;

    forwardKinematics(model_handler_.getModel(), data_, q);
    updateFramePlacements(model_handler_.getModel(), data_);
    computeCentroidalMomentum(model_handler_.getModel(), data_, q, v);

    if (updateJacobians)
    {
      updateJacobiansMassMatrix(x);
    }
  }

  void RobotDataHandler::updateJacobiansMassMatrix(const ConstVectorRef & x)
  {
    const Eigen::Block q = x.head(model_handler_.getModel().nq);
    const Eigen::Block v = x.tail(model_handler_.getModel().nv);

    computeJointJacobians(model_handler_.getModel(), data_);
    computeJointJacobiansTimeVariation(model_handler_.getModel(), data_, q, v);
    crba(model_handler_.getModel(), data_, q);
    data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
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
