/// @copyright Copyright (C) 2024 INRIA
#include "simple-mpc/config.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc::python {

namespace bp = boost::python;

/* FORWARD DECLARATIONS */
void exposeHandler();
void exposeOcpHandler();
void exposeFullDynamicsProblem();
void exposeCentroidalProblem();
void exposeKinodynamicsProblem();
void exposeMPC();
void exposeIDSolver();
void exposeIKIDSolver();

/* PYTHON MODULE */
BOOST_PYTHON_MODULE(simple_mpc_pywrap) {
  bp::import("pinocchio");
  bp::import("aligator");
  bp::scope().attr("__version__") = SIMPLE_MPC_VERSION;
  ENABLE_SPECIFIC_MATRIX_TYPE(Eigen::VectorXi);
  exposeHandler();
  exposeOcpHandler();
  exposeFullDynamicsProblem();
  exposeCentroidalProblem();
  exposeKinodynamicsProblem();
  exposeMPC();
  exposeIDSolver();
  exposeIKIDSolver();
}

} // namespace simple_mpc::python
