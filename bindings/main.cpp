#include <eigenpy/eigenpy.hpp>

#include "simple-mpc/python.hpp"

BOOST_PYTHON_MODULE(simple_mpc_pywrap) {
  boost::python::import("pinocchio");
  boost::python::import("aligator");
  // Enabling eigenpy support, i.e. numpy/eigen compatibility.
  eigenpy::enableEigenPy();
  ENABLE_SPECIFIC_MATRIX_TYPE(Eigen::VectorXi);
  simple_mpc::python::exposeHandler();
  simple_mpc::python::exposeBaseProblem();
  simple_mpc::python::exposeFullDynamicsProblem();
  simple_mpc::python::exposeCentroidalProblem();
  simple_mpc::python::exposeKinodynamicsProblem();
  simple_mpc::python::exposeMPC();
}
