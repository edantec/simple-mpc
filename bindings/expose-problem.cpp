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
#include <eigenpy/std-map.hpp>
#include <eigenpy/std-vector.hpp>
#include <fmt/format.h>
#include <pinocchio/bindings/python/utils/pickle-map.hpp>
#include <pinocchio/fwd.hpp>

#include "problems.hpp"
#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/fulldynamics.hpp"

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

void exposeBaseProblem() {
  bp::register_ptr_to_python<std::shared_ptr<Problem>>();
  bp::class_<PyProblem, boost::noncopyable>("Problem", bp::no_init)
      .def(bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("createStage", bp::pure_virtual(&Problem::createStage),
           bp::args("self", "contact_map", "force_refs"))
      .def("createTerminalCost", bp::pure_virtual(&Problem::createTerminalCost),
           bp::args("self"))
      .def("createTerminalConstraint",
           bp::pure_virtual(&Problem::createTerminalConstraint),
           bp::args("self"))
      .def("updateTerminalConstraint",
           bp::pure_virtual(&Problem::updateTerminalConstraint),
           bp::args("self"))
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
      .def("getProblemState", bp::pure_virtual(&Problem::getProblemState),
           bp::args("self"))
      .def("createProblem", &Problem::createProblem,
           bp::args("self", "x0", "horizon", "force_size", "gravity"))
      .def("setReferenceControl", &Problem::setReferenceControl,
           bp::args("self", "t", "u_ref"))
      .def("getReferenceControl", &Problem::getReferenceControl,
           bp::args("self", "t"))
      .def("getProblem", &Problem::getProblem, bp::args("self"));
}

void initializeFull(FullDynamicsProblem &self, const bp::dict &settings) {
  FullDynamicsSettings conf;
  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.u0 = bp::extract<Eigen::VectorXd>(settings["u0"]);
  conf.DT = bp::extract<double>(settings["DT"]);
  conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);
  conf.w_forces = bp::extract<Eigen::MatrixXd>(settings["w_forces"]);
  conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);
  /// Foot parameters
  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  conf.umin = bp::extract<Eigen::VectorXd>(settings["umin"]);
  conf.umax = bp::extract<Eigen::VectorXd>(settings["umax"]);

  conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
  conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

  self.initialize(conf);
}

StageModel createFullStage(FullDynamicsProblem &self,
                           const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict) {
  boost::python::list phase_keys = boost::python::list(phase_dict.keys());
  boost::python::list pose_keys = boost::python::list(pose_dict.keys());
  boost::python::list force_keys = boost::python::list(force_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
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

  return self.createStage(phase_contact, pose_contact, force_contact);
}

bp::dict getSettingsFull(FullDynamicsProblem &self) {
  FullDynamicsSettings conf = self.getSettings();
  bp::dict settings;
  settings["x0"] = conf.x0;
  settings["u0"] = conf.u0;
  settings["DT"] = conf.DT;
  settings["w_x"] = conf.w_x;
  settings["w_u"] = conf.w_u;
  settings["w_cent"] = conf.w_cent;
  settings["gravity"] = conf.gravity;
  settings["force_size"] = conf.force_size;
  settings["w_forces"] = conf.w_forces;
  settings["w_frame"] = conf.w_frame;
  settings["umin"] = conf.umin;
  settings["umax"] = conf.umax;
  settings["qmin"] = conf.qmin;
  settings["qmax"] = conf.qmax;
  settings["mu"] = conf.mu;
  settings["Lfoot"] = conf.Lfoot;
  settings["Wfoot"] = conf.Wfoot;

  return settings;
}

void createFullProblem(FullDynamicsProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity) {

  self.createProblem(x0, horizon, force_size, gravity);
}

TrajOptProblem getFullProblem(FullDynamicsProblem &self) {
  return *self.getProblem();
}

void exposeFullDynamicsProblem() {
  boost::python::register_ptr_to_python<std::shared_ptr<FullDynamicsProblem>>();
  StdVectorPythonVisitor<std::vector<ContactMap>, true>::expose(
      "StdVec_ContactMap_double");

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

  bp::class_<PyFullDynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "FullDynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initializeFull, bp::args("self", "settings"))
      .def("getSettings", &getSettingsFull)
      .def("initialize",
           bp::make_function(
               &FullDynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createFullStage)
      .def("createProblem", &createFullProblem)
      .def("setReferencePose", &FullDynamicsProblem::setReferencePose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("setReferencePoses", &FullDynamicsProblem::setReferencePoses,
           bp::args("self", "t", "pose_refs"))
      .def("setTerminalReferencePose",
           &FullDynamicsProblem::setTerminalReferencePose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("setReferenceForces", &FullDynamicsProblem::setReferenceForces,
           bp::args("self", "t", "force_refs"))
      .def("setReferenceForce", &FullDynamicsProblem::setReferenceForce,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("getReferencePose", &FullDynamicsProblem::getReferencePose,
           bp::args("self", "t", "cost_name"))
      .def("getReferenceForce", &FullDynamicsProblem::getReferenceForce,
           bp::args("self", "t", "cost_name"))
      .def("getProblemState", &FullDynamicsProblem::getProblemState,
           bp::args("self"))
      .def("createTerminalCost", &FullDynamicsProblem::createTerminalCost,
           bp::args("self"))
      .def("updateTerminalConstraint",
           &FullDynamicsProblem::updateTerminalConstraint, bp::args("self"))
      .def("createTerminalConstraint",
           &FullDynamicsProblem::createTerminalConstraint, bp::args("self"))
      .def("getProblem", &getFullProblem);
}

void initializeCent(CentroidalProblem &self, const bp::dict &settings) {
  CentroidalSettings conf;
  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.u0 = bp::extract<Eigen::VectorXd>(settings["u0"]);
  conf.DT = bp::extract<double>(settings["DT"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_linear_mom = bp::extract<Eigen::Matrix3d>(settings["w_linear_mom"]);
  conf.w_angular_mom = bp::extract<Eigen::Matrix3d>(settings["w_angular_mom"]);
  conf.w_linear_acc = bp::extract<Eigen::Matrix3d>(settings["w_linear_acc"]);
  conf.w_angular_acc = bp::extract<Eigen::Matrix3d>(settings["w_angular_acc"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);

  self.initialize(conf);
}

StageModel createCentStage(CentroidalProblem &self, const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict) {
  boost::python::list phase_keys = boost::python::list(phase_dict.keys());
  boost::python::list pose_keys = boost::python::list(pose_dict.keys());
  boost::python::list force_keys = boost::python::list(force_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
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

  return self.createStage(phase_contact, pose_contact, force_contact);
}

bp::dict getSettingsCent(CentroidalProblem &self) {
  CentroidalSettings conf = self.getSettings();
  bp::dict settings;
  settings["x0"] = conf.x0;
  settings["u0"] = conf.u0;
  settings["DT"] = conf.DT;
  settings["w_u"] = conf.w_u;
  settings["w_linear_mom"] = conf.w_linear_mom;
  settings["w_angular_mom"] = conf.w_angular_mom;
  settings["w_linear_acc"] = conf.w_linear_acc;
  settings["w_angular_acc"] = conf.w_angular_acc;
  settings["gravity"] = conf.gravity;
  settings["force_size"] = conf.force_size;

  return settings;
}

void createCentProblem(CentroidalProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity) {

  self.createProblem(x0, horizon, force_size, gravity);
}

TrajOptProblem getCentProblem(FullDynamicsProblem &self) {
  return *self.getProblem();
}

void exposeCentroidalProblem() {
  boost::python::register_ptr_to_python<std::shared_ptr<CentroidalProblem>>();

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

  bp::class_<PyCentroidalProblem, bp::bases<Problem>, boost::noncopyable>(
      "CentroidalProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initializeCent, bp::args("self", "settings"))
      .def("getSettings", &getSettingsCent)
      .def("initialize",
           bp::make_function(
               &CentroidalProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createCentStage)
      .def("createProblem", &createCentProblem)
      .def("setReferencePose", &CentroidalProblem::setReferencePose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("setReferencePoses", &CentroidalProblem::setReferencePoses,
           bp::args("self", "t", "pose_refs"))
      .def("setTerminalReferencePose",
           &CentroidalProblem::setTerminalReferencePose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("setReferenceForces", &CentroidalProblem::setReferenceForces,
           bp::args("self", "t", "force_refs"))
      .def("setReferenceForce", &CentroidalProblem::setReferenceForce,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("getReferencePose", &CentroidalProblem::getReferencePose,
           bp::args("self", "t", "cost_name"))
      .def("getReferenceForce", &CentroidalProblem::getReferenceForce,
           bp::args("self", "t", "cost_name"))
      .def("getProblemState", &CentroidalProblem::getProblemState,
           bp::args("self"))
      .def("createTerminalCost", &CentroidalProblem::createTerminalCost,
           bp::args("self"))
      .def("createTerminalConstraint",
           &CentroidalProblem::createTerminalConstraint, bp::args("self"))
      .def("updateTerminalConstraint",
           &CentroidalProblem::updateTerminalConstraint, bp::args("self"))
      .def("getProblem", &getCentProblem);
}

void initializeKino(KinodynamicsProblem &self, const bp::dict &settings) {
  KinodynamicsSettings conf;
  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.u0 = bp::extract<Eigen::VectorXd>(settings["u0"]);
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

  self.initialize(conf);
}

bp::dict getSettingsKino(KinodynamicsProblem &self) {
  KinodynamicsSettings conf = self.getSettings();
  bp::dict settings;
  settings["x0"] = conf.x0;
  settings["u0"] = conf.u0;
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

  return settings;
}

StageModel createKinoStage(KinodynamicsProblem &self,
                           const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict) {
  boost::python::list phase_keys = boost::python::list(phase_dict.keys());
  boost::python::list pose_keys = boost::python::list(pose_dict.keys());
  boost::python::list force_keys = boost::python::list(force_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
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

  return self.createStage(phase_contact, pose_contact, force_contact);
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
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
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
      .def("setReferencePoses", &KinodynamicsProblem::setReferencePoses,
           bp::args("self", "t", "pose_refs"))
      .def("setTerminalReferencePose",
           &KinodynamicsProblem::setTerminalReferencePose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("setReferenceForces", &KinodynamicsProblem::setReferenceForces,
           bp::args("self", "t", "force_refs"))
      .def("setReferenceForce", &KinodynamicsProblem::setReferenceForce,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("getReferencePose", &KinodynamicsProblem::getReferencePose,
           bp::args("self", "t", "cost_name"))
      .def("getReferenceForce", &KinodynamicsProblem::getReferenceForce,
           bp::args("self", "t", "cost_name"))
      .def("getProblemState", &KinodynamicsProblem::getProblemState,
           bp::args("self"))
      .def("createTerminalCost", &KinodynamicsProblem::createTerminalCost,
           bp::args("self"))
      .def("createTerminalConstraint",
           &KinodynamicsProblem::createTerminalConstraint, bp::args("self"))
      .def("updateTerminalConstraint",
           &KinodynamicsProblem::updateTerminalConstraint, bp::args("self"))
      .def("getProblem", &getKinoProblem);
}
} // namespace python
} // namespace simple_mpc
