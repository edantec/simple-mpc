///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <aligator/modelling/contact-map.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-map.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/python/py-ocp-handler.hpp"

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;
using eigenpy::python::StdMapPythonVisitor;
using ContactMap = ContactMapTpl<double>;

void exposeContainers() {
  StdMapPythonVisitor<
      std::string, Eigen::VectorXd, std::less<std::string>,
      std::allocator<std::pair<const std::string, Eigen::VectorXd>>,
      true>::expose("StdMap_Force");

  StdMapPythonVisitor<
      std::string, pinocchio::SE3, std::less<std::string>,
      std::allocator<std::pair<const std::string, pinocchio::SE3>>,
      true>::expose("StdMap_SE3");

  StdMapPythonVisitor<std::string, bool, std::less<std::string>,
                      std::allocator<std::pair<const std::string, bool>>,
                      true>::expose("StdMap_bool");

  StdVectorPythonVisitor<std::vector<ContactMap>, true>::expose(
      "StdVec_ContactMap_double");
}

void exposeBaseProblem() {
  bp::register_ptr_to_python<std::shared_ptr<Problem>>();
  bp::class_<PyProblem, boost::noncopyable>("Problem", bp::no_init)
      .def(bp::init<const RobotHandler &>(("self"_a, "robot_handler")))
      .def("createStage", bp::pure_virtual(&Problem::createStage),
           ("self"_a, "contact_map", "force_refs", "land_constraint"))
      .def("createTerminalCost", bp::pure_virtual(&Problem::createTerminalCost),
           "self"_a)
      .def("createTerminalConstraint",
           bp::pure_virtual(&Problem::createTerminalConstraint), "self"_a)
      .def("updateTerminalConstraint",
           bp::pure_virtual(&Problem::updateTerminalConstraint),
           ("self"_a, "com_ref"))
      .def("setReferencePose", bp::pure_virtual(&Problem::setReferencePose),
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("setReferencePoses", bp::pure_virtual(&Problem::setReferencePoses),
           bp::args("self", "t", "pose_refs"))
      .def("setTerminalReferencePose",
           bp::pure_virtual(&Problem::setTerminalReferencePose),
           bp::args("self", "ee_name", "pose_ref"))
      .def("getReferencePose", bp::pure_virtual(&Problem::getReferencePose),
           bp::args("self", "t", "ee_name"))
      .def("setReferenceForces", bp::pure_virtual(&Problem::setReferenceForces),
           bp::args("self", "t", "force_refs"))
      .def("setReferenceForce", bp::pure_virtual(&Problem::setReferenceForce),
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("getReferenceForce", bp::pure_virtual(&Problem::getReferenceForce),
           bp::args("self", "t", "ee_name"))
      .def("setVelocityBase", bp::pure_virtual(&Problem::setVelocityBase),
           bp::args("self", "t", "velocity_base"))
      .def("getVelocityBase", bp::pure_virtual(&Problem::getVelocityBase),
           bp::args("self", "t"))
      .def("setPoseBase", bp::pure_virtual(&Problem::setPoseBase),
           bp::args("self", "t", "pose_base"))
      .def("getPoseBase", bp::pure_virtual(&Problem::getPoseBase),
           bp::args("self", "t"))
      .def("getProblemState", bp::pure_virtual(&Problem::getProblemState),
           bp::args("self"))
      .def("getContactSupport", bp::pure_virtual(&Problem::getContactSupport),
           bp::args("self", "t"))
      .def("createProblem", &Problem::createProblem,
           ("self"_a, "x0", "horizon", "force_size", "gravity",
            "terminal_constraint"))
      .def("setReferenceControl", &Problem::setReferenceControl,
           ("self"_a, "t", "u_ref"))
      .def("getReferenceControl", &Problem::getReferenceControl,
           ("self"_a, "t"))
      .def("getProblem", &Problem::getProblem, "self"_a);

  exposeContainers();
}

} // namespace python
} // namespace simple_mpc
