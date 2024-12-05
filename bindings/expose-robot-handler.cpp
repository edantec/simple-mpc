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

void exposeHandler() {
    bp::class_<RobotModelHandler>("RobotModelHandler", bp::init<>())
        .def("addFrameToBase", bp::make_function(&RobotModelHandler::addFrameToBase))
        .def("difference", bp::make_function(&RobotModelHandler::difference))
        .def("shapeState", bp::make_function(&RobotModelHandler::shapeState))
        .def("getRootFrameId", bp::make_function(&RobotModelHandler::getRootFrameId))
        .def("getFootIndex", bp::make_function(&RobotModelHandler::getFootIndex))
        .def("getFootName", bp::make_function(&RobotModelHandler::getFootName))
        .def("getFeetNames", bp::make_function(&RobotModelHandler::getFeetNames))
        .def("getFootId", bp::make_function(&RobotModelHandler::getFootId))
        .def("getRefFootId", bp::make_function(&RobotModelHandler::getRefFootId))
        .def("getMass", bp::make_function(&RobotModelHandler::getMass))
        .def("getModel", bp::make_function(&RobotModelHandler::getModel))
        .def("getCompleteModel", bp::make_function(&RobotModelHandler::getCompleteModel))


    bp::class_<RobotDataHandler>("RobotDataHandler", bp::init<>())
        .def("updateInternalData", bp::make_function(&RobotDataHandler::updateInternalData))
        .def("updateJacobiansMassMatrix", bp::make_function(&RobotDataHandler::updateJacobiansMassMatrix))
        .def("getRefFootPose", bp::make_function(&RobotDataHandler::getRefFootPose))
        .def("getFootPose", bp::make_function(&RobotDataHandler::getFootPose))
        .def("getRootFramePose", bp::make_function(&RobotDataHandler::getRootFramePose))
        .def("getModelHandler", bp::make_function(&RobotDataHandler::getModelHandler))
        .def("getData", bp::make_function(&RobotDataHandler::getData))
        .def("getCentroidalState", bp::make_function(&RobotDataHandler::getCentroidalState))
  return;
}

} // namespace python
} // namespace simple_mpc
