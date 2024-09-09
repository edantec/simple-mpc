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
      .def("create_stage", bp::pure_virtual(&Problem::create_stage),
           bp::args("self", "contact_map", "force_refs"))
      .def("create_terminal_cost",
           bp::pure_virtual(&Problem::create_terminal_cost), bp::args("self"))
      .def("set_reference_pose", bp::pure_virtual(&Problem::set_reference_pose),
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("set_reference_poses",
           bp::pure_virtual(&Problem::set_reference_poses),
           bp::args("self", "t", "pose_refs"))
      .def("set_terminal_reference_pose",
           bp::pure_virtual(&Problem::set_terminal_reference_pose),
           bp::args("self", "ee_name", "pose_ref"))
      .def("get_reference_pose", bp::pure_virtual(&Problem::get_reference_pose),
           bp::args("self", "t", "ee_name"))
      .def("set_reference_forces",
           bp::pure_virtual(&Problem::set_reference_forces),
           bp::args("self", "t", "force_refs"))
      .def("set_reference_force",
           bp::pure_virtual(&Problem::set_reference_force),
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("get_reference_force",
           bp::pure_virtual(&Problem::get_reference_force),
           bp::args("self", "t", "ee_name"))
      .def("get_x0_from_multibody",
           bp::pure_virtual(&Problem::get_x0_from_multibody),
           bp::args("self", "x_multibody"))
      .def("create_problem", &Problem::create_problem,
           bp::args("self", "x0", "horizon", "force_size", "gravity"))
      .def("set_reference_control", &Problem::set_reference_control,
           bp::args("self", "t", "u_ref"))
      .def("get_reference_control", &Problem::get_reference_control,
           bp::args("self", "t"))
      .def("get_problem", &Problem::get_problem, bp::args("self"))
      .def("get_cost_map", &Problem::get_cost_map, bp::args("self"));
}

void initialize_full(FullDynamicsProblem &self, const bp::dict &settings) {
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

bp::dict get_settings_full(FullDynamicsProblem &self) {
  FullDynamicsSettings conf = self.get_settings();
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

StageModel create_full_stage(FullDynamicsProblem &self,
                             const ContactMap &contact_map,
                             const bp::dict &force_dict) {
  boost::python::list keys = boost::python::list(force_dict.keys());
  std::map<std::string, Eigen::VectorXd> force_refs;
  for (int i = 0; i < len(keys); ++i) {
    boost::python::extract<std::string> extractor(keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_refs.insert({key, ff});
    }
  }

  return self.create_stage(contact_map, force_refs);
}

void create_full_problem(FullDynamicsProblem &self, const Eigen::VectorXd &x0,
                         const size_t horizon, const int force_size,
                         const double gravity) {

  self.create_problem(x0, horizon, force_size, gravity);
}

TrajOptProblem get_full_problem(FullDynamicsProblem &self) {
  return *self.get_problem();
}

void exposeFullDynamicsProblem() {
  boost::python::register_ptr_to_python<std::shared_ptr<FullDynamicsProblem>>();
  StdVectorPythonVisitor<std::vector<ContactMap>, true>::expose(
      "StdVec_ContactMap_double");

  eigenpy::python::StdMapPythonVisitor<
      std::string, Eigen::VectorXd, std::less<std::string>,
      std::allocator<std::pair<const std::string, Eigen::VectorXd>>,
      true>::expose("StdMap_Force");

  bp::class_<PyFullDynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "FullDynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initialize_full, bp::args("self", "settings"))
      .def("get_settings", &get_settings_full)
      .def("initialize",
           bp::make_function(
               &FullDynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("create_stage", &create_full_stage)
      .def("create_problem", &create_full_problem)
      .def("set_reference_pose", &FullDynamicsProblem::set_reference_pose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("set_reference_poses", &FullDynamicsProblem::set_reference_poses,
           bp::args("self", "t", "pose_refs"))
      .def("set_terminal_reference_pose",
           &FullDynamicsProblem::set_terminal_reference_pose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("set_reference_forces", &FullDynamicsProblem::set_reference_forces,
           bp::args("self", "t", "force_refs"))
      .def("set_reference_force", &FullDynamicsProblem::set_reference_force,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("get_reference_pose", &FullDynamicsProblem::get_reference_pose,
           bp::args("self", "t", "cost_name"))
      .def("get_reference_force", &FullDynamicsProblem::get_reference_force,
           bp::args("self", "t", "cost_name"))
      .def("get_x0_from_multibody", &FullDynamicsProblem::get_x0_from_multibody,
           bp::args("self", "x_multibody"))
      .def("create_terminal_cost", &FullDynamicsProblem::create_terminal_cost,
           bp::args("self"))
      .def("get_problem", &get_full_problem)
      .def("get_cost_map", &Problem::get_cost_map, bp::args("self"));
}

void initialize_cent(CentroidalProblem &self, const bp::dict &settings) {
  CentroidalSettings conf;
  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.u0 = bp::extract<Eigen::VectorXd>(settings["u0"]);
  conf.DT = bp::extract<double>(settings["DT"]);
  conf.w_x_ter = bp::extract<Eigen::MatrixXd>(settings["w_x_ter"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_linear_mom = bp::extract<Eigen::Matrix3d>(settings["w_linear_mom"]);
  conf.w_angular_mom = bp::extract<Eigen::Matrix3d>(settings["w_angular_mom"]);
  conf.w_linear_acc = bp::extract<Eigen::Matrix3d>(settings["w_linear_acc"]);
  conf.w_angular_acc = bp::extract<Eigen::Matrix3d>(settings["w_angular_acc"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);

  self.initialize(conf);
}

bp::dict get_settings_cent(CentroidalProblem &self) {
  CentroidalSettings conf = self.get_settings();
  bp::dict settings;
  settings["x0"] = conf.x0;
  settings["u0"] = conf.u0;
  settings["DT"] = conf.DT;
  settings["w_x_ter"] = conf.w_x_ter;
  settings["w_u"] = conf.w_u;
  settings["w_linear_mom"] = conf.w_linear_mom;
  settings["w_angular_mom"] = conf.w_angular_mom;
  settings["w_linear_acc"] = conf.w_linear_acc;
  settings["w_angular_acc"] = conf.w_angular_acc;
  settings["gravity"] = conf.gravity;
  settings["force_size"] = conf.force_size;

  return settings;
}

StageModel create_cent_stage(CentroidalProblem &self,
                             const ContactMap &contact_map,
                             const bp::dict &force_dict) {
  boost::python::list keys = boost::python::list(force_dict.keys());
  std::map<std::string, Eigen::VectorXd> force_refs;
  for (int i = 0; i < len(keys); ++i) {
    boost::python::extract<std::string> extractor(keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_refs.insert({key, ff});
    }
  }

  return self.create_stage(contact_map, force_refs);
}

void create_cent_problem(CentroidalProblem &self, const Eigen::VectorXd &x0,
                         const size_t horizon, const int force_size,
                         const double gravity) {

  self.create_problem(x0, horizon, force_size, gravity);
}

TrajOptProblem get_cent_problem(FullDynamicsProblem &self) {
  return *self.get_problem();
}

void exposeCentroidalProblem() {
  boost::python::register_ptr_to_python<std::shared_ptr<CentroidalProblem>>();

  bp::class_<PyCentroidalProblem, bp::bases<Problem>, boost::noncopyable>(
      "CentroidalProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initialize_cent, bp::args("self", "settings"))
      .def("get_settings", &get_settings_cent)
      .def("initialize",
           bp::make_function(
               &CentroidalProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("create_stage", &create_cent_stage)
      .def("create_problem", &create_cent_problem)
      .def("set_reference_pose", &CentroidalProblem::set_reference_pose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("set_reference_poses", &CentroidalProblem::set_reference_poses,
           bp::args("self", "t", "pose_refs"))
      .def("set_terminal_reference_pose",
           &CentroidalProblem::set_terminal_reference_pose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("set_reference_forces", &CentroidalProblem::set_reference_forces,
           bp::args("self", "t", "force_refs"))
      .def("set_reference_force", &CentroidalProblem::set_reference_force,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("get_reference_pose", &CentroidalProblem::get_reference_pose,
           bp::args("self", "t", "cost_name"))
      .def("get_reference_force", &CentroidalProblem::get_reference_force,
           bp::args("self", "t", "cost_name"))
      .def("get_x0_from_multibody", &CentroidalProblem::get_x0_from_multibody,
           bp::args("self", "x_multibody"))
      .def("create_terminal_cost", &CentroidalProblem::create_terminal_cost,
           bp::args("self"))
      .def("get_problem", &get_cent_problem)
      .def("get_cost_map", &Problem::get_cost_map, bp::args("self"));
}

void initialize_kino(KinodynamicsProblem &self, const bp::dict &settings) {
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

  self.initialize(conf);
}

bp::dict get_settings_kino(KinodynamicsProblem &self) {
  KinodynamicsSettings conf = self.get_settings();
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

  return settings;
}

StageModel create_kino_stage(KinodynamicsProblem &self,
                             const ContactMap &contact_map,
                             const bp::dict &force_dict) {
  boost::python::list keys = boost::python::list(force_dict.keys());
  std::map<std::string, Eigen::VectorXd> force_refs;
  for (int i = 0; i < len(keys); ++i) {
    boost::python::extract<std::string> extractor(keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_refs.insert({key, ff});
    }
  }

  return self.create_stage(contact_map, force_refs);
}

void create_kino_problem(KinodynamicsProblem &self, const Eigen::VectorXd &x0,
                         const size_t horizon, const int force_size,
                         const double gravity) {

  self.create_problem(x0, horizon, force_size, gravity);
}

TrajOptProblem get_kino_problem(KinodynamicsProblem &self) {
  return *self.get_problem();
}

void exposeKinodynamicsProblem() {
  boost::python::register_ptr_to_python<
      boost::shared_ptr<PyKinodynamicsProblem>>();
  boost::python::register_ptr_to_python<
      boost::shared_ptr<KinodynamicsProblem>>();

  bp::class_<PyKinodynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "KinodynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initialize_kino, bp::args("self", "settings"))
      .def("get_settings", &get_settings_kino)
      .def("initialize",
           bp::make_function(
               &KinodynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("create_stage", &create_kino_stage)
      .def("create_problem", &create_kino_problem)
      .def("set_reference_pose", &KinodynamicsProblem::set_reference_pose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("set_reference_poses", &KinodynamicsProblem::set_reference_poses,
           bp::args("self", "t", "pose_refs"))
      .def("set_terminal_reference_pose",
           &KinodynamicsProblem::set_terminal_reference_pose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("set_reference_forces", &KinodynamicsProblem::set_reference_forces,
           bp::args("self", "t", "force_refs"))
      .def("set_reference_force", &KinodynamicsProblem::set_reference_force,
           bp::args("self", "t", "ee_name", "force_ref"))
      .def("get_reference_pose", &KinodynamicsProblem::get_reference_pose,
           bp::args("self", "t", "cost_name"))
      .def("get_reference_force", &KinodynamicsProblem::get_reference_force,
           bp::args("self", "t", "cost_name"))
      .def("get_x0_from_multibody", &KinodynamicsProblem::get_x0_from_multibody,
           bp::args("self", "x_multibody"))
      .def("create_terminal_cost", &KinodynamicsProblem::create_terminal_cost,
           bp::args("self"))
      .def("get_problem", &get_kino_problem)
      .def("get_cost_map", &Problem::get_cost_map, bp::args("self"));
}
} // namespace python
} // namespace simple_mpc
