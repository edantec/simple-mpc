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

#include "simple-mpc/python/py-ocp-handler.hpp"

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;
using eigenpy::python::StdMapPythonVisitor;

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

void exposeOcpHandler() {
  bp::register_ptr_to_python<std::shared_ptr<OCPHandler>>();
  bp::class_<PyOCPHandler, boost::noncopyable>("OCPHandler", bp::no_init)
      .def(bp::init<const RobotModelHandler &, const RobotDataHandler&>(("self", "model_handler", "data_handler")))
      .def("createStage", bp::pure_virtual(&OCPHandler::createStage),
           ("self"_a, "contact_map", "force_refs", "land_constraint"))
      .def("createTerminalCost",
           bp::pure_virtual(&OCPHandler::createTerminalCost), "self"_a)
      .def("createTerminalConstraint",
           bp::pure_virtual(&OCPHandler::createTerminalConstraint), "self"_a)
      .def("updateTerminalConstraint",
           &CentroidalProblem::updateTerminalConstraint,
           bp::args("self", "com_ref"))
      .def("getProblem", &getCentProblem);
}

void initializeKino(KinodynamicsProblem &self, const bp::dict &settings) {
  KinodynamicsSettings conf;
  conf.DT = bp::extract<double>(settings["DT"]);
  conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);
  conf.w_centder = bp::extract<Eigen::MatrixXd>(settings["w_centder"]);
  conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);

  conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
  conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
  conf.force_cone = bp::extract<bool>(settings["force_cone"]);

  self.initialize(conf);
}

bp::dict getSettingsKino(KinodynamicsProblem &self) {
  KinodynamicsSettings conf = self.getSettings();
  bp::dict settings;
  settings["DT"] = conf.DT;
  settings["w_x"] = conf.w_x;
  settings["w_u"] = conf.w_u;
  settings["w_cent"] = conf.w_cent;
  settings["w_centder"] = conf.w_centder;
  settings["w_frame"] = conf.w_frame;
  settings["gravity"] = conf.gravity;
  settings["force_size"] = conf.force_size;
  settings["qmin"] = conf.qmin;
  settings["qmax"] = conf.qmax;
  settings["mu"] = conf.mu;
  settings["Lfoot"] = conf.Lfoot;
  settings["Wfoot"] = conf.Wfoot;
  settings["kinematics_limits"] = conf.kinematics_limits;
  settings["force_cone"] = conf.force_cone;

  return settings;
}

StageModel createKinoStage(KinodynamicsProblem &self,
                           const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict,
                           const bp::dict &land_dict) {
  boost::python::list phase_keys = boost::python::list(phase_dict.keys());
  boost::python::list pose_keys = boost::python::list(pose_dict.keys());
  boost::python::list force_keys = boost::python::list(force_dict.keys());
  boost::python::list land_keys = boost::python::list(land_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
  std::map<std::string, bool> land_constraint;
  for (int i = 0; i < len(phase_keys); ++i) {
    boost::python::extract<std::string> extractor(phase_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(phase_dict[key]);
      phase_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(pose_keys); ++i) {
    boost::python::extract<std::string> extractor(pose_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
      pose_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(force_keys); ++i) {
    boost::python::extract<std::string> extractor(force_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(land_keys); ++i) {
    boost::python::extract<std::string> extractor(land_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(land_dict[key]);
      land_constraint.insert({key, ff});
    }
  }

  return self.createStage(phase_contact, pose_contact, force_contact,
                          land_constraint);
}

void createKinoProblem(KinodynamicsProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity) {

  self.createProblem(x0, horizon, force_size, gravity);
}

TrajOptProblem getKinoProblem(KinodynamicsProblem &self) {
  return *self.getProblem();
}

void exposeKinodynamicsProblem() {
  boost::python::register_ptr_to_python<
      boost::shared_ptr<PyKinodynamicsProblem>>();
  boost::python::register_ptr_to_python<
      boost::shared_ptr<KinodynamicsProblem>>();

  eigenpy::python::StdMapPythonVisitor<
      std::string, Eigen::VectorXd, std::less<std::string>,
      std::allocator<std::pair<const std::string, Eigen::VectorXd>>,
      true>::expose("StdMap_Force");

  eigenpy::python::StdMapPythonVisitor<
      std::string, pinocchio::SE3, std::less<std::string>,
      std::allocator<std::pair<const std::string, pinocchio::SE3>>,
      true>::expose("StdMap_SE3");

  eigenpy::python::StdMapPythonVisitor<
      std::string, bool, std::less<std::string>,
      std::allocator<std::pair<const std::string, bool>>,
      true>::expose("StdMap_bool");

  bp::class_<PyKinodynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "KinodynamicsProblem",
      bp::init<const RobotModelHandler &, const RobotDataHandler&>(bp::args("self", "model_handler", "data_handler")))
      .def("initialize", &initializeKino, bp::args("self", "settings"))
      .def("getSettings", &getSettingsKino)
      .def("initialize",
           bp::make_function(
               &KinodynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createKinoStage)
      .def("createProblem", &createKinoProblem)
      .def("setReferencePose", &KinodynamicsProblem::setReferencePose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("setReferencePoses",
           bp::pure_virtual(&OCPHandler::setReferencePoses),
           bp::args("self", "t", "pose_refs"))
      .def("setTerminalReferencePose",
           bp::pure_virtual(&OCPHandler::setTerminalReferencePose),
           bp::args("self", "ee_name", "pose_ref"))
      .def("getReferencePose", bp::pure_virtual(&OCPHandler::getReferencePose),
           bp::args("self", "t", "ee_name"))
      .def("setReferenceForces",
           bp::pure_virtual(&OCPHandler::setReferenceForces),
           bp::args("self", "t", "force_refs"))
      .def("setReferenceForce",
           bp::pure_virtual(&OCPHandler::setReferenceForce),
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("getReferenceForce",
           bp::pure_virtual(&OCPHandler::getReferenceForce),
           bp::args("self", "t", "ee_name"))
      .def("setVelocityBase", bp::pure_virtual(&OCPHandler::setVelocityBase),
           bp::args("self", "t", "velocity_base"))
      .def("getVelocityBase", bp::pure_virtual(&OCPHandler::getVelocityBase),
           bp::args("self", "t"))
      .def("setPoseBase", bp::pure_virtual(&OCPHandler::setPoseBase),
           bp::args("self", "t", "pose_base"))
      .def("getPoseBase", bp::pure_virtual(&OCPHandler::getPoseBase),
           bp::args("self", "t"))
      .def("getProblemState", bp::pure_virtual(&OCPHandler::getProblemState),
           bp::args("self"))
      .def("getContactSupport",
           bp::pure_virtual(&OCPHandler::getContactSupport),
           bp::args("self", "t"))
      .def("createProblem", &OCPHandler::createProblem,
           ("self"_a, "x0", "horizon", "force_size", "gravity",
            "terminal_constraint"))
      .def("setReferenceControl", &OCPHandler::setReferenceControl,
           ("self"_a, "t", "u_ref"))
      .def("getReferenceControl", &OCPHandler::getReferenceControl,
           ("self"_a, "t"))
      .def(
          "getProblem",
          +[](OCPHandler &ocp) { return boost::ref(ocp.getProblem()); },
          "self"_a);

  exposeContainers();
}

} // namespace python
} // namespace simple_mpc
