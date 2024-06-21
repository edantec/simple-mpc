/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/python.hpp"

#include "simple-mpc/base-problem.hpp"
#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/kinodynamics.hpp"

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
struct PyProblem : Problem, bp::wrapper<Problem> {
  using Problem::Problem;

  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(aligator::StageModelTpl<double>,
                                    "create_stage", contact_map, force_refs);
  }

  CostStack create_terminal_cost() override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(aligator::CostStackTpl<double>,
                                    "create_terminal_cost");
  }

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &force_sequence) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "create_problem", x0,
                                    contact_sequence, force_sequence);
  }

  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "set_reference_poses", t, pose_refs);
  }

  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(pinocchio::SE3, "get_reference_pose", t,
                                    ee_name);
  }

  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "set_reference_forces", t,
                                    force_refs);
  }

  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "set_reference_force", t, ee_name,
                                    force_ref);
  }

  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "get_reference_force", t,
                                    ee_name);
  }

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override {
    SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "get_x0_from_multibody",
                                    x_multibody);
  }

  void set_reference_control(const std::size_t t,
                             const Eigen::VectorXd &u_ref) {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, Problem, set_reference_control, t, u_ref);
  }

  Eigen::VectorXd get_reference_control(const std::size_t t) {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, Problem, get_reference_control,
                               t);
  }
};

struct PyFullDynamicsProblem : FullDynamicsProblem,
                               bp::wrapper<FullDynamicsProblem> {
  using FullDynamicsProblem::FullDynamicsProblem;

  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::StageModelTpl<double>,
                               FullDynamicsProblem, create_stage, contact_map,
                               force_refs);
  }

  CostStack create_terminal_cost() override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::CostStackTpl<double>,
                               FullDynamicsProblem, create_terminal_cost);
  }

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &force_sequence) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, FullDynamicsProblem, create_problem, x0,
                               contact_sequence, force_sequence);
  }

  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, FullDynamicsProblem, set_reference_poses,
                               t, pose_refs);
  }

  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(pinocchio::SE3, FullDynamicsProblem,
                               get_reference_pose, t, ee_name);
  }

  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, FullDynamicsProblem, set_reference_forces,
                               t, force_refs);
  }

  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, FullDynamicsProblem, set_reference_force,
                               t, ee_name, force_ref);
  }

  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, FullDynamicsProblem,
                               get_reference_force, t, ee_name);
  }

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, FullDynamicsProblem,
                               get_x0_from_multibody, x_multibody);
  }
};

struct PyCentroidalProblem : CentroidalProblem, bp::wrapper<CentroidalProblem> {
  using CentroidalProblem::CentroidalProblem;

  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::StageModelTpl<double>,
                               CentroidalProblem, create_stage, contact_map,
                               force_refs);
  }

  CostStack create_terminal_cost() override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::CostStackTpl<double>,
                               CentroidalProblem, create_terminal_cost);
  }

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &force_sequence) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, CentroidalProblem, create_problem, x0,
                               contact_sequence, force_sequence);
  }

  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, CentroidalProblem, set_reference_poses, t,
                               pose_refs);
  }

  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(pinocchio::SE3, CentroidalProblem,
                               get_reference_pose, t, ee_name);
  }

  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, CentroidalProblem, set_reference_forces, t,
                               force_refs);
  }

  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, CentroidalProblem, set_reference_force, t,
                               ee_name, force_ref);
  }

  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, CentroidalProblem,
                               get_reference_force, t, ee_name);
  }

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, CentroidalProblem,
                               get_x0_from_multibody, x_multibody);
  }
};

struct PyKinodynamicsProblem : KinodynamicsProblem,
                               bp::wrapper<KinodynamicsProblem> {
  using KinodynamicsProblem::KinodynamicsProblem;

  StageModel create_stage(
      const ContactMap &contact_map,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::StageModelTpl<double>,
                               KinodynamicsProblem, create_stage, contact_map,
                               force_refs);
  }

  CostStack create_terminal_cost() override {
    SIMPLE_MPC_PYTHON_OVERRIDE(aligator::CostStackTpl<double>,
                               KinodynamicsProblem, create_terminal_cost);
  }

  void create_problem(const Eigen::VectorXd &x0,
                      const std::vector<ContactMap> &contact_sequence,
                      const std::vector<std::map<std::string, Eigen::VectorXd>>
                          &force_sequence) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, KinodynamicsProblem, create_problem, x0,
                               contact_sequence, force_sequence);
  }

  void set_reference_poses(
      const std::size_t t,
      const std::map<std::string, pinocchio::SE3> &pose_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, KinodynamicsProblem, set_reference_poses,
                               t, pose_refs);
  }

  pinocchio::SE3 get_reference_pose(const std::size_t t,
                                    const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(pinocchio::SE3, KinodynamicsProblem,
                               get_reference_pose, t, ee_name);
  }

  void set_reference_forces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, KinodynamicsProblem, set_reference_forces,
                               t, force_refs);
  }

  void set_reference_force(const std::size_t t, const std::string &ee_name,
                           const Eigen::VectorXd &force_ref) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(void, KinodynamicsProblem, set_reference_force,
                               t, ee_name, force_ref);
  }

  Eigen::VectorXd get_reference_force(const std::size_t t,
                                      const std::string &ee_name) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, KinodynamicsProblem,
                               get_reference_force, t, ee_name);
  }

  Eigen::VectorXd
  get_x0_from_multibody(const Eigen::VectorXd &x_multibody) override {
    SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, KinodynamicsProblem,
                               get_x0_from_multibody, x_multibody);
  }
};

} // namespace python
} // namespace simple_mpc
