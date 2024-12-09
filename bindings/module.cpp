#include "simple-mpc/config.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc::python {

namespace bp = boost::python;

BOOST_PYTHON_MODULE(simple_mpc_pywrap) {
  bp::import("pinocchio");
  bp::import("aligator");
  bp::scope().attr("__version__") = SIMPLE_MPC_VERSION;
  ENABLE_SPECIFIC_MATRIX_TYPE(Eigen::VectorXi);
  exposeHandler();
  exposeBaseProblem();
  exposeFullDynamicsProblem();
  exposeCentroidalProblem();
  exposeKinodynamicsProblem();
  exposeMPC();
  exposeIDSolver();
  exposeIKIDSolver();
}

}
