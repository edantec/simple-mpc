#pragma once

#include <pinocchio/fwd.hpp>

#include <boost/python.hpp>

namespace simple_mpc {
namespace python {

void exposeHandler();
void exposeBaseProblem();
void exposeFullDynamicsProblem();

} // namespace python
} // namespace simple_mpc
