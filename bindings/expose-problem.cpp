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
#include <crocoddyl/core/activation-base.hpp>
#include <eigenpy/eigenpy.hpp>
#include <fmt/format.h>
#include <pinocchio/fwd.hpp>
#include <simple-mpc/base-problem.hpp>
#include <simple-mpc/fulldynamics.hpp>
#include <type_traits>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;

namespace internal {
template <typename ret_type>
ret_type suppress_if_void(boost::python::detail::method_result &&o) {
  if constexpr (!std::is_void_v<ret_type>) {
    return o;
  } else {
    return;
  }
}
} // namespace internal

#define SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, pyname, ...)                 \
  do {                                                                         \
    if (bp::override fo = this->get_override(pyname)) {                        \
      decltype(auto) o = fo(__VA_ARGS__);                                      \
      return ::simple_mpc::python::internal::suppress_if_void<ret_type>(       \
          std::move(o));                                                       \
    }                                                                          \
  } while (false)

/**
 * @def ALIGATOR_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)
 * @brief Define the body of a virtual function override. This is meant
 *        to reduce boilerplate code when exposing virtual member functions.
 */
#define SIMPLE_MPC_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)                 \
  SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, pyname, __VA_ARGS__);              \
  throw std::runtime_error("Tried to call pure virtual function");

/**
 * @def ALIGATOR_PYTHON_OVERRIDE(ret_type, cname, fname, ...)
 * @copybrief ALIGATOR_PYTHON_OVERRIDE_PURE()
 */
#define SIMPLE_MPC_PYTHON_OVERRIDE(ret_type, cname, fname, ...)                \
  SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, #fname, __VA_ARGS__);              \
  return cname::fname(__VA_ARGS__);

template <typename T>
inline void py_list_to_std_vector(const bp::object &iterable,
                                  std::vector<T> &out) {
  out = std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                       boost::python::stl_input_iterator<T>());
}

template <class T> bp::list std_vector_to_py_list(const std::vector<T> &v) {
  bp::object get_iter = bp::iterator<std::vector<T>>();
  bp::object iter = get_iter(v);
  bp::list l(iter);
  return l;
}

/* struct PyProblem : Problem, bp::wrapper<Problem> {
  StageModel
  create_stage(const ContactMap &contact_map,
               const std::map<std::string, Eigen::VectorXd> &force_refs) const
override { SIMPLE_MPC_PYTHON_OVERRIDE_PURE(StageModel, "create_stage",
contact_map, force_refs,);
  }

  CostStack create_terminal_cost() {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "create_terminal_cost",);
  }

  void
  create_problem(const Eigen::VectorXd &x0,
                 const std::vector<ContactMap> &contact_sequence,
                 const std::vector<std::map<std::string, Eigen::VectorXd>>
                     &force_sequence) {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, "create_problem", x0, contact_sequence,
force_sequence);
  }

  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, "set_reference_poses", t, pose_refs,);
  }

  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) {
    SIMPLE_MPC_PYTHON_OVERRIDE(pinocchio::SE3, "get_reference_pose", t,
ee_name,);
  }

  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, "set_reference_forces", t, force_refs,);
  }

  void set_reference_force(const std::size_t t,
                           const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, "set_reference_force", t, ee_name,
force_ref,);
  }

  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, "get_reference_force", t,
ee_name,);
  }

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, "get_x0_from_multibody",
x_multibody,);
  }
}; */

/* void exposeBaseProblem() {
  bp::class_<Problem>("Problem", bp::no_init)
  .def(bp::init<const RobotHandler &>(
    bp::args("self", "handler")));
} */

void initialize(FullDynamicsProblem &self, const bp::dict &settings) {
  FullDynamicsSettings conf;
  conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
  conf.u0 = bp::extract<Eigen::VectorXd>(settings["u0"]);
  conf.DT = bp::extract<double>(settings["DT"]);

  conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);

  conf.w_forces = bp::extract<Eigen::MatrixXd>(settings["w_forces"]);
  conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

  conf.umin = bp::extract<Eigen::VectorXd>(settings["umin"]);
  conf.umax = bp::extract<Eigen::VectorXd>(settings["umax"]);

  conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
  conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

  /// Foot parameters
  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  self.initialize(conf);
}

bp::dict get_settings(FullDynamicsProblem &self) {
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

void exposeFullDynamicsProblem() {
  bp::class_<FullDynamicsProblem, bp::bases<Problem>>(
      "FullDynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initialize, bp::args("self", "settings"))
      .def("get_settings", &get_settings)
      .def("initialize",
           bp::make_function(
               &FullDynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("create_problem", &FullDynamicsProblem::create_problem,
           bp::args("self", "x0", "contact_sequence", "force_sequence"))
      .def("create_stage", &FullDynamicsProblem::create_stage,
           bp::args("self", "contact_map", "force_refs"))
      .def("set_reference_poses", &FullDynamicsProblem::set_reference_poses,
           bp::args("self", "t", "pose_refs"))
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
           bp::args("self"));
}
} // namespace python
} // namespace simple_mpc
