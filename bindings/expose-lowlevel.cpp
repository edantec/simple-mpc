///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <eigenpy/eigenpy.hpp>
#include <fmt/format.h>
#include <pinocchio/fwd.hpp>

#include "simple-mpc/lowlevel-control.hpp"
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace simple_mpc {
namespace python {
namespace bp = boost::python;

template <typename T>
inline void py_list_to_std_vector(const bp::object &iterable,
                                  std::vector<T> &out) {
  out = std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                       boost::python::stl_input_iterator<T>());
}

void initialize_ID(IDSolver &self, const bp::dict &settings,
                   const pinocchio::Model &model) {
  IDSettings conf;

  py_list_to_std_vector(settings["contact_ids"], conf.contact_ids);

  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  conf.force_size = bp::extract<int>(settings["force_size"]);
  conf.kd = bp::extract<double>(settings["kd"]);
  conf.w_force = bp::extract<double>(settings["w_force"]);
  conf.w_acc = bp::extract<double>(settings["w_acc"]);
  conf.verbose = bp::extract<bool>(settings["verbose"]);

  self.initialize(conf, model);
}

void initialize_IKID(IKIDSolver &self, const bp::dict &settings,
                     const pinocchio::Model &model) {
  IKIDSettings conf;

  py_list_to_std_vector(settings["Kp_gains"], conf.Kp_gains);
  py_list_to_std_vector(settings["Kd_gains"], conf.Kd_gains);
  py_list_to_std_vector(settings["contact_ids"], conf.contact_ids);
  py_list_to_std_vector(settings["fixed_frame_ids"], conf.fixed_frame_ids);

  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.dt = bp::extract<double>(settings["dt"]);
  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  conf.force_size = bp::extract<int>(settings["force_size"]);
  conf.w_qref = bp::extract<double>(settings["w_qref"]);
  conf.w_footpose = bp::extract<double>(settings["w_footpose"]);
  conf.w_centroidal = bp::extract<double>(settings["w_centroidal"]);
  conf.w_baserot = bp::extract<double>(settings["w_baserot"]);
  conf.w_force = bp::extract<double>(settings["w_force"]);
  conf.verbose = bp::extract<bool>(settings["verbose"]);

  self.initialize(conf, model);
}

void exposeIDSolver() {
  bp::class_<IDSolver>("IDSolver", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def("initialize", &initialize_ID)
      .def("solve_qp", &IDSolver::solve_qp,
           bp::args("self", "data", "contact_state", "v", "a", "forces", "M"))
      .def("getQP", &IDSolver::getQP, bp::args("self"),
           bp::return_internal_reference<>(), "Get QP solver.")
      .add_property("solved_acc", &IDSolver::solved_acc_)
      .add_property("solved_forces", &IDSolver::solved_forces_)
      .add_property("solved_torque", &IDSolver::solved_torque_);
  ;
}

void exposeIKIDSolver() {
  bp::class_<IKIDSolver>("IKIDSolver", bp::no_init)
      .def(bp::init<>(bp::args("self")))
      .def("initialize", &initialize_IKID)
      .def("solve_qp", &IKIDSolver::solve_qp,
           bp::args("self", "data", "contact_state", "x_measured", "forces",
                    "foot_refs", "foot_refs_next", "dH", "M"))
      .add_property("solved_acc", &IKIDSolver::solved_acc_)
      .add_property("solved_forces", &IKIDSolver::solved_forces_)
      .add_property("solved_torque", &IKIDSolver::solved_torque_);
}

} // namespace python
} // namespace simple_mpc
