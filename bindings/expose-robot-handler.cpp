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

  self.initialize(conf);
}

bp::dict get_settings(RobotHandler &self) {
  RobotHandlerSettings conf = self.get_settings();
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
      .def("get_settings", &get_settings)
      .def("updateInternalData", &RobotHandler::updateInternalData)
      .def("shapeState",
           bp::make_function(
               &RobotHandler::shapeState,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("set_q0", &RobotHandler::set_q0)
      .def("get_rmodel",
           bp::make_function(
               &RobotHandler::get_rmodel,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_rdata",
           bp::make_function(
               &RobotHandler::get_rdata,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_com_position",
           bp::make_function(
               &RobotHandler::get_com_position,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_q0",
           bp::make_function(
               &RobotHandler::get_q0,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_v0",
           bp::make_function(
               &RobotHandler::get_v0,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_x0",
           bp::make_function(
               &RobotHandler::get_x0,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_settings",
           bp::make_function(
               &RobotHandler::get_settings,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("get_controlledJointsIDs",
           bp::make_function(
               &RobotHandler::get_controlledJointsIDs,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("get_ee_names",
           bp::make_function(
               &RobotHandler::get_ee_names,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("get_ee_pose",
           bp::make_function(
               &RobotHandler::get_ee_pose,
               bp::return_value_policy<bp::copy_const_reference>()))
      .def("get_mass",
           bp::make_function(
               &RobotHandler::get_mass,
               bp::return_value_policy<bp::copy_const_reference>()));

  return;
}

} // namespace python
} // namespace simple_mpc
