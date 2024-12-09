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
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/python/py-ocp-handler.hpp"

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;
using ContactMap = ContactMapTpl<double>;
void exposeBaseProblem() {
  bp::register_ptr_to_python<std::shared_ptr<Problem>>();
  bp::class_<PyProblem, boost::noncopyable>("Problem", bp::no_init)
      .def(bp::init<const RobotHandler &>(("self"_a, "handler")))
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
           bp::args("self", "x0", "horizon", "force_size", "gravity",
                    "terminal_constraint"))
      .def("setReferenceControl", &Problem::setReferenceControl,
           bp::args("self", "t", "u_ref"))
      .def("getReferenceControl", &Problem::getReferenceControl,
           bp::args("self", "t"))
      .def("getProblem", &Problem::getProblem, "self"_a);
}

void initializeFull(FullDynamicsProblem &self, const bp::dict &settings) {
  FullDynamicsSettings conf;
  conf.timestep = bp::extract<double>(settings["timestep"]);
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

  /// Limits
  conf.umin = bp::extract<Eigen::VectorXd>(settings["umin"]);
  conf.umax = bp::extract<Eigen::VectorXd>(settings["umax"]);

  conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
  conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

  /// Baumgarte correctors
  conf.Kp_correction = bp::extract<Eigen::VectorXd>(settings["Kp_correction"]);
  conf.Kd_correction = bp::extract<Eigen::VectorXd>(settings["Kd_correction"]);

  /// Constraints
  conf.torque_limits = bp::extract<bool>(settings["torque_limits"]);
  conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
  conf.force_cone = bp::extract<bool>(settings["force_cone"]);

  self.initialize(conf);
}

StageModel createFullStage(FullDynamicsProblem &self,
                           const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict,
                           const bp::dict &land_dict) {
  bp::list phase_keys(phase_dict.keys());
  bp::list pose_keys(pose_dict.keys());
  bp::list force_keys(force_dict.keys());
  bp::list land_keys(land_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
  std::map<std::string, bool> land_constraint;
  for (int i = 0; i < len(phase_keys); ++i) {
    bp::extract<std::string> extractor(phase_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(phase_dict[key]);
      phase_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(pose_keys); ++i) {
    bp::extract<std::string> extractor(pose_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
      pose_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(force_keys); ++i) {
    bp::extract<std::string> extractor(force_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(land_keys); ++i) {
    bp::extract<std::string> extractor(land_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(land_dict[key]);
      land_constraint.insert({key, ff});
    }
  }

  return self.createStage(phase_contact, pose_contact, force_contact,
                          land_constraint);
}

bp::dict getSettingsFull(FullDynamicsProblem &self) {
  FullDynamicsSettings conf = self.getSettings();
  bp::dict settings;
  settings["timestep"] = conf.timestep;
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
  settings["Kp_correction"] = conf.Kp_correction;
  settings["Kd_correction"] = conf.Kd_correction;
  settings["mu"] = conf.mu;
  settings["Lfoot"] = conf.Lfoot;
  settings["Wfoot"] = conf.Wfoot;
  settings["torque_limits"] = conf.torque_limits;
  settings["kinematics_limits"] = conf.kinematics_limits;
  settings["force_cone"] = conf.force_cone;

  return settings;
}

void createFullProblem(FullDynamicsProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity, const bool terminal_constraint) {

  self.createProblem(x0, horizon, force_size, gravity, terminal_constraint);
}

void exposeFullDynamicsProblem() {
  bp::register_ptr_to_python<std::shared_ptr<FullDynamicsProblem>>();
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

  bp::class_<FullDynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "FullDynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initializeFull, bp::args("self", "settings"))
      .def("getSettings", &getSettingsFull)
      .def("initialize",
           bp::make_function(
               &FullDynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createFullStage)
      .def("createProblem", &createFullProblem);
}

void initializeCent(CentroidalProblem &self, const bp::dict &settings) {
  CentroidalSettings conf;
  conf.timestep = bp::extract<double>(settings["timestep"]);
  conf.w_com = bp::extract<Eigen::Matrix3d>(settings["w_com"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_linear_mom = bp::extract<Eigen::Matrix3d>(settings["w_linear_mom"]);
  conf.w_angular_mom = bp::extract<Eigen::Matrix3d>(settings["w_angular_mom"]);
  conf.w_linear_acc = bp::extract<Eigen::Matrix3d>(settings["w_linear_acc"]);
  conf.w_angular_acc = bp::extract<Eigen::Matrix3d>(settings["w_angular_acc"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);

  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  self.initialize(conf);
}

StageModel createCentStage(CentroidalProblem &self, const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict,
                           const bp::dict &land_dict) {
  bp::list phase_keys = bp::list(phase_dict.keys());
  bp::list pose_keys = bp::list(pose_dict.keys());
  bp::list force_keys = bp::list(force_dict.keys());
  bp::list land_keys = bp::list(land_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
  std::map<std::string, bool> land_constraint;
  for (int i = 0; i < len(phase_keys); ++i) {
    bp::extract<std::string> extractor(phase_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(phase_dict[key]);
      phase_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(pose_keys); ++i) {
    bp::extract<std::string> extractor(pose_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
      pose_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(force_keys); ++i) {
    bp::extract<std::string> extractor(force_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(land_keys); ++i) {
    bp::extract<std::string> extractor(land_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(land_dict[key]);
      land_constraint.insert({key, ff});
    }
  }

  return self.createStage(phase_contact, pose_contact, force_contact,
                          land_constraint);
}

bp::dict getSettingsCent(CentroidalProblem &self) {
  CentroidalSettings conf = self.getSettings();
  bp::dict settings;
  settings["timestep"] = conf.timestep;
  settings["w_com"] = conf.w_com;
  settings["w_u"] = conf.w_u;
  settings["w_linear_mom"] = conf.w_linear_mom;
  settings["w_angular_mom"] = conf.w_angular_mom;
  settings["w_linear_acc"] = conf.w_linear_acc;
  settings["w_angular_acc"] = conf.w_angular_acc;
  settings["gravity"] = conf.gravity;
  settings["mu"] = conf.mu;
  settings["Lfoot"] = conf.Lfoot;
  settings["Wfoot"] = conf.Wfoot;
  settings["force_size"] = conf.force_size;

  return settings;
}

void createCentProblem(CentroidalProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity, const bool terminal_constraint) {

  self.createProblem(x0, horizon, force_size, gravity, terminal_constraint);
}

void exposeCentroidalProblem() {
  bp::register_ptr_to_python<std::shared_ptr<CentroidalProblem>>();

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

  bp::class_<CentroidalProblem, bp::bases<Problem>, boost::noncopyable>(
      "CentroidalProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initializeCent, bp::args("self", "settings"))
      .def("getSettings", &getSettingsCent)
      .def("initialize",
           bp::make_function(
               &CentroidalProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createCentStage)
      .def("createProblem", &createCentProblem);
}

void initializeKino(KinodynamicsProblem &self, const bp::dict &settings) {
  KinodynamicsSettings conf;
  conf.timestep = bp::extract<double>(settings["timestep"]);
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
  settings["timestep"] = conf.timestep;
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
  bp::list phase_keys = bp::list(phase_dict.keys());
  bp::list pose_keys = bp::list(pose_dict.keys());
  bp::list force_keys = bp::list(force_dict.keys());
  bp::list land_keys = bp::list(land_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
  std::map<std::string, bool> land_constraint;
  for (int i = 0; i < len(phase_keys); ++i) {
    bp::extract<std::string> extractor(phase_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(phase_dict[key]);
      phase_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(pose_keys); ++i) {
    bp::extract<std::string> extractor(pose_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
      pose_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(force_keys); ++i) {
    bp::extract<std::string> extractor(force_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(land_keys); ++i) {
    bp::extract<std::string> extractor(land_keys[i]);
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
                       const double gravity, const bool terminal_constraint) {

  self.createProblem(x0, horizon, force_size, gravity, terminal_constraint);
}

void exposeKinodynamicsProblem() {
  bp::register_ptr_to_python<shared_ptr<KinodynamicsProblem>>();

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

  bp::class_<KinodynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
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
      .def("createTerminalConstraint",
           &KinodynamicsProblem::createTerminalConstraint, bp::args("self"))
      .def("updateTerminalConstraint",
           &KinodynamicsProblem::updateTerminalConstraint,
           bp::args("self", "com_ref"));
}
} // namespace python
} // namespace simple_mpc
