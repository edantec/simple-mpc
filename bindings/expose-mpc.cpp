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
#include <eigenpy/std-map.hpp>
#include <eigenpy/std-vector.hpp>
#include <fmt/format.h>
#include <pinocchio/bindings/python/utils/pickle-map.hpp>
#include <pinocchio/fwd.hpp>

#include "simple-mpc/mpc.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

void initialize(MPC &self, const bp::dict &settings,
                std::shared_ptr<Problem> problem) {
  MPCSettings conf;

  conf.totalSteps = bp::extract<int>(settings["totalSteps"]);
  conf.ddpIteration = bp::extract<int>(settings["ddpIteration"]);

  conf.min_force = bp::extract<double>(settings["min_force"]);
  conf.support_force = bp::extract<double>(settings["support_force"]);

  conf.TOL = bp::extract<double>(settings["TOL"]);
  conf.mu_init = bp::extract<double>(settings["mu_init"]);
  conf.max_iters = bp::extract<std::size_t>(settings["max_iters"]);
  conf.num_threads = bp::extract<std::size_t>(settings["max_iters"]);

  conf.swing_apex = bp::extract<double>(settings["swing_apex"]);
  conf.x_translation = bp::extract<double>(settings["x_translation"]);
  conf.y_translation = bp::extract<double>(settings["y_translation"]);
  conf.T_fly = bp::extract<int>(settings["T_fly"]);
  conf.T_contact = bp::extract<int>(settings["T_contact"]);
  conf.T = bp::extract<std::size_t>(settings["T"]);

  self.initialize(conf, problem);
}

void exposeMPC() {
  using StageVec = std::vector<StageModel>;
  using MapBool = std::map<std::string, bool>;
  StdVectorPythonVisitor<StageVec, true>::expose(
      "StdVec_StageModel",
      eigenpy::details::overload_base_get_item_for_std_vector<StageVec>());

  eigenpy::python::StdMapPythonVisitor<
      std::string, bool, std::less<std::string>,
      std::allocator<std::pair<const std::string, bool>>,
      true>::expose("StdMap_Bool");

  StdVectorPythonVisitor<std::vector<MapBool>, true>::expose("StdVec_MapBool");

  bp::class_<MPC>("MPC", bp::no_init)
      .def(bp::init<const Eigen::VectorXd &, const Eigen::VectorXd &>(
          bp::args("self", "x_multibody", "u0")))
      .def("initialize", &initialize)
      .def("generateFullHorizon", &MPC::generateFullHorizon,
           bp::args("self", "contact_states"))
      .def("iterate", &MPC::iterate, bp::args("self", "q_current", "v_current"))
      .def("setReferencePose", &MPC::setReferencePose,
           bp::args("self", "t", "ee_name", "pose_ref"))
      .def("setTerminalReferencePose", &MPC::setTerminalReferencePose,
           bp::args("self", "ee_name", "pose_ref"))
      .def("get_fullHorizon", &MPC::get_fullHorizon, bp::args("self"),
           bp::return_internal_reference<>(), "Get the full horizon.")
      .def("get_foot_takeoff_timings", &MPC::get_foot_takeoff_timings,
           bp::args("self", "ee_name"), bp::return_internal_reference<>(),
           "Get the takeoff timings.")
      .def("get_foot_land_timings", &MPC::get_foot_land_timings,
           bp::args("self", "ee_name"), bp::return_internal_reference<>(),
           "Get the land timings.")
      .def("get_handler", &MPC::get_handler, bp::args("self"),
           bp::return_internal_reference<>(), "Get the robot handler.")
      .add_property("xs", &MPC::xs_)
      .add_property("us", &MPC::us_)
      .add_property("foot_takeoff_times", &MPC::foot_takeoff_times_)
      .add_property("foot_land_times", &MPC::foot_land_times_)
      .add_property("K0", &MPC::K0_)
      .add_property("horizon_iteration", &MPC::horizon_iteration_);
}

} // namespace python
} // namespace simple_mpc
