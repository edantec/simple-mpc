#include "simple-mpc/model-utils.hpp"

#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/context.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

void makeTalosReduced(std::vector<std::string> controlled_joints_names, Model & model, Eigen::VectorXd & q0)
{
  const std::string talos_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
  const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";

  pinocchio::Model model_complete;
  pin::urdf::buildModel(talos_path, pin::JointModelFreeFlyer(), model_complete);
  pin::srdf::loadReferenceConfigurations(model_complete, srdf_path, false);
  Eigen::VectorXd q0_complete = model_complete.referenceConfigurations["half_sitting"];

  // Check if listed joints belong to model
  for (std::vector<std::string>::const_iterator it = controlled_joints_names.begin();
       it != controlled_joints_names.end(); ++it)
  {
    const std::string & joint_name = *it;
    // std::cout << joint_name << std::endl;
    // std::cout << model_complete.getJointId(joint_name) << std::endl;
    if (not(model_complete.existJointName(joint_name)))
    {
      std::cout << "joint: " << joint_name << " does not belong to the model" << std::endl;
    }
  }

  // making list of blocked joints
  std::vector<unsigned long> locked_joints_id;
  for (std::vector<std::string>::const_iterator it = model_complete.names.begin() + 1; it != model_complete.names.end();
       ++it)
  {
    const std::string & joint_name = *it;
    if (
      std::find(controlled_joints_names.begin(), controlled_joints_names.end(), joint_name)
      == controlled_joints_names.end())
    {
      locked_joints_id.push_back(model_complete.getJointId(joint_name));
    }
  }
  model = pin::buildReducedModel(model_complete, locked_joints_id, q0_complete);
  q0 = model.referenceConfigurations["half_sitting"];
}
