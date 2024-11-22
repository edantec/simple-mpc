///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <eigenpy/eigenpy.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;

template <typename T>
inline void py_list_to_std_vector(const bp::object &iterable,
                                  std::vector<T> &out) {
  out = std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                       boost::python::stl_input_iterator<T>());
}

void initialize(RobotHandler &self, bp::dict settings) {
  RobotHandlerSettings conf;
  conf.urdf_path = bp::extract<std::string>(settings["urdf_path"]);
  conf.srdf_path = bp::extract<std::string>(settings["srdf_path"]);
  conf.robot_description =
      bp::extract<std::string>(settings["robot_description"]);
  conf.root_name = bp::extract<std::string>(settings["root_name"]);
  conf.base_configuration =
      bp::extract<std::string>(settings["base_configuration"]);
  py_list_to_std_vector(settings["controlled_joints_names"],
                        conf.controlled_joints_names);
  py_list_to_std_vector(settings["end_effector_names"],
                        conf.end_effector_names);
  py_list_to_std_vector(settings["hip_names"], conf.hip_names);
  py_list_to_std_vector(settings["feet_to_base_trans"], conf.feet_to_base_trans);
  self.initialize(conf);
}

bp::dict getSettings(RobotHandler &self) {
  RobotHandlerSettings conf = self.getSettings();
  bp::dict settings;
  settings["urdfPath"] = conf.urdf_path;
  settings["srdfPath"] = conf.srdf_path;
  settings["robot_description"] = conf.robot_description;
  settings["root_name"] = conf.root_name;
  settings["base_configuration"] = conf.base_configuration;

  return settings;
}

void exposeHandler() {
  bp::class_<RobotHandler>("RobotHandler", bp::init<>())
      .def("initialize", &initialize)
      .def("getSettings", &getSettings)
      .def("updateConfiguration", &RobotHandler::updateConfiguration)
      .def("updateState", &RobotHandler::updateState)
      .def("updateInternalData", &RobotHandler::updateInternalData)
      .def("updateJacobiansMassMatrix",
           &RobotHandler::updateJacobiansMassMatrix)
      .def("shapeState", &RobotHandler::shapeState)
      .def("difference", &RobotHandler::difference)
      .def("getModel",
           bp::make_function(
               &RobotHandler::getModel,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getCompleteModel",
           bp::make_function(
               &RobotHandler::getCompleteModel,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getData",
           bp::make_function(
               &RobotHandler::getData,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getComPosition",
           bp::make_function(
               &RobotHandler::getComPosition,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getConfiguration",
           bp::make_function(
               &RobotHandler::getConfiguration,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getCompleteConfiguration",
           bp::make_function(
               &RobotHandler::getCompleteConfiguration,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getVelocity",
           bp::make_function(
               &RobotHandler::getVelocity,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getState",
           bp::make_function(
               &RobotHandler::getState,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getCentroidalState",
           bp::make_function(
               &RobotHandler::getCentroidalState,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getSettings",
           bp::make_function(
               &RobotHandler::getSettings,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("getControlledJointsIDs",
           bp::make_function(
               &RobotHandler::getControlledJointsIDs,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getRootId",
           bp::make_function(
               &RobotHandler::getRootId,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getFootId",
           bp::make_function(
               &RobotHandler::getFootId,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getFeetNames",
           bp::make_function(
               &RobotHandler::getFeetNames,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getFeetIds",
           bp::make_function(
               &RobotHandler::getFeetIds,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getFootPose",
           bp::make_function(
               &RobotHandler::getFootPose,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("getMass", bp::make_function(
                          &RobotHandler::getMass,
                          bp::return_value_policy<bp::copy_const_reference>()))
      .def("getMassMatrix",
           bp::make_function(
               &RobotHandler::getMassMatrix,
               bp::return_value_policy<bp::copy_const_reference>()));

  return;
}

} // namespace python
} // namespace simple_mpc
