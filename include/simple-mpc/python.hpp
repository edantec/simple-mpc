#pragma once

#include <pinocchio/fwd.hpp>

#include <boost/python.hpp>

namespace simple_mpc {
namespace python {

void exposeHandler();
void exposeBaseProblem();
void exposeFullDynamicsProblem();
void exposeCentroidalProblem();
void exposeKinodynamicsProblem();
void exposeMPC();
void exposeIDSolver();
void exposeIKIDSolver();

} // namespace python
} // namespace simple_mpc
