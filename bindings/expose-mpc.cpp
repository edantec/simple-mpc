///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/bindings/python/utils/pickle-map.hpp>
#include <pinocchio/fwd.hpp>

#include <eigenpy/deprecation-policy.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/mpc.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;
    using eigenpy::StdVectorPythonVisitor;

    void initialize(MPC & self, const bp::dict & settings, std::shared_ptr<OCPHandler> problem)
    {
      MPCSettings conf;

      conf.ddpIteration = bp::extract<int>(settings["ddpIteration"]);

      conf.support_force = bp::extract<double>(settings["support_force"]);

      conf.TOL = bp::extract<double>(settings["TOL"]);
      conf.mu_init = bp::extract<double>(settings["mu_init"]);
      conf.max_iters = bp::extract<std::size_t>(settings["max_iters"]);
      conf.num_threads = bp::extract<std::size_t>(settings["num_threads"]);

      conf.swing_apex = bp::extract<double>(settings["swing_apex"]);
      conf.T_fly = bp::extract<int>(settings["T_fly"]);
      conf.T_contact = bp::extract<int>(settings["T_contact"]);
      conf.timestep = bp::extract<double>(settings["timestep"]);

      self.initialize(conf, problem);
    }

    bp::dict getSettings(MPC & self)
    {
      MPCSettings & conf = self.settings_;
      bp::dict settings;
      settings["ddpIteration"] = conf.ddpIteration;
      settings["support_force"] = conf.support_force;
      settings["TOL"] = conf.TOL;
      settings["mu_init"] = conf.mu_init;
      settings["max_iters"] = conf.max_iters;
      settings["num_threads"] = conf.num_threads;
      settings["swing_apex"] = conf.swing_apex;
      settings["T_fly"] = conf.T_fly;
      settings["T_contact"] = conf.T_contact;
      settings["timestep"] = conf.timestep;

      return settings;
    }

    void exposeMPC()
    {
      using StageVec = std::vector<std::shared_ptr<StageModel>>;
      using MapBool = std::map<std::string, bool>;
      StdVectorPythonVisitor<StageVec, true>::expose(
        "StdVec_StageModel", eigenpy::details::overload_base_get_item_for_std_vector<StageVec>());

      StdVectorPythonVisitor<std::vector<MapBool>, true>::expose("StdVec_MapBool");

      bp::class_<MPC>("MPC", bp::no_init)
        .def(bp::init<>(bp::args("self")))
        .def("initialize", &initialize)
        .def("getSettings", &getSettings)
        .def("generateCycleHorizon", &MPC::generateCycleHorizon, bp::args("self", "contact_states"))
        .def("iterate", &MPC::iterate, bp::args("self", "x"))
        .def("setReferencePose", &MPC::setReferencePose, bp::args("self", "t", "ee_name", "pose_ref"))
        .def("getReferencePose", &MPC::getReferencePose, bp::args("self", "t", "ee_name"))
        .def("setTerminalReferencePose", &MPC::setTerminalReferencePose, bp::args("self", "ee_name", "pose_ref"))
        .def_readwrite("velocity_base", &MPC::velocity_base_)
        .def_readwrite("pose_base", &MPC::pose_base_)
        .def_readonly("ocp_handler", &MPC::ocp_handler_)
        .def("setPoseBase", &MPC::setPoseBase, ("self"_a, "pose_base"))
        .def("getPoseBase", &MPC::getPoseBase, ("self"_a, "t"))
        .def("switchToWalk", &MPC::switchToWalk, ("self"_a, "velocity_base"))
        .def("switchToStand", &MPC::switchToStand, "self"_a)
        .def("getFootTakeoffCycle", &MPC::getFootTakeoffCycle, ("self"_a, "ee_name"))
        .def("getFootLandCycle", &MPC::getFootLandCycle, ("self"_a, "ee_name"))
        .def("getCyclingContactState", &MPC::getCyclingContactState, ("self"_a, "t", "ee_name"))
        .def(
          "getModelHandler", &MPC::getModelHandler, "self"_a, bp::return_internal_reference<>(),
          "Get the robot model handler.")
        .def(
          "getDataHandler", &MPC::getDataHandler, "self"_a, bp::return_internal_reference<>(),
          "Get the robot data handler.")
        .def(
          "getTrajOptProblem", &MPC::getTrajOptProblem, "self"_a, bp::return_internal_reference<>(),
          "Get the trajectory optimal problem.")
        .def(
          "getCycleHorizon", &MPC::getCycleHorizon, "self"_a, bp::return_internal_reference<>(),
          "Get the cycle horizon.")
        .def(
          "getSolver", &MPC::getSolver, "self"_a,
          eigenpy::deprecated_member<eigenpy::DeprecationType::DEPRECATION, bp::return_internal_reference<>>(),
          "Get the SolverProxDDP object")
        .def_readonly("solver", &MPC::solver_)
        .add_property("xs", &MPC::xs_)
        .add_property("us", &MPC::us_)
        .add_property("Ks", &MPC::Ks_);
    }

  } // namespace python
} // namespace simple_mpc
