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
#include <eigenpy/std-vector.hpp>
#include <fmt/format.h>
#include <pinocchio/fwd.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/mpc.hpp"

namespace simple_mpc {
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

/* void setup(MPC &self, FullDynamicsProblem &problem) {
  std::shared_ptr<Problem> pb_ptr =
std::make_shared<FullDynamicsProblem>(problem); self.setup(pb_ptr);
} */

void initialize(MPC &self, const bp::dict &settings,
                std::shared_ptr<Problem> problem) {
  MPCSettings conf;

  conf.totalSteps = bp::extract<int>(settings["totalSteps"]);
  conf.T = bp::extract<std::size_t>(settings["T"]);
  conf.ddpIteration = bp::extract<int>(settings["ddpIteration"]);

  conf.min_force = bp::extract<double>(settings["min_force"]);
  conf.support_force = bp::extract<double>(settings["support_force"]);

  conf.TOL = bp::extract<double>(settings["TOL"]);
  conf.mu_init = bp::extract<double>(settings["mu_init"]);
  conf.max_iters = bp::extract<std::size_t>(settings["max_iters"]);
  conf.num_threads = bp::extract<std::size_t>(settings["max_iters"]);

  self.initialize(conf, problem);
}

void exposeMPC() {
  StdVectorPythonVisitor<std::vector<StageModel>, true>::expose(
      "StdVec_StageModel_double");

  bp::class_<MPC>("MPC", bp::no_init)
      .def(bp::init<const Eigen::VectorXd &, const Eigen::VectorXd &>(
          bp::args("self", "x_multibody", "u0")))
      .def("initialize", &initialize)
      .def("generateFullHorizon", &MPC::generateFullHorizon,
           bp::args("self", "contact_phases", "contact_forces"))
      .def("iterate", &MPC::iterate, bp::args("self", "q_current", "v_current"))
      .add_property("xs", &MPC::xs_)
      .add_property("us", &MPC::us_)
      .add_property("foot_takeoff_times", &MPC::foot_takeoff_times_)
      .add_property("foot_land_times", &MPC::foot_land_times_)
      .add_property("K0", &MPC::K0_)
      .add_property("horizon_iteration", &MPC::horizon_iteration_);
}

} // namespace python
} // namespace simple_mpc
