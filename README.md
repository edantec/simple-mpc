# Simple-mpc

**Simple-mpc** is a C++ implementation of multiple predictive control schemes for locomotion based on the Aligator optimization solver.

It can be used with quadrupeds and bipeds to generate whole-body walking motions based on a pre-defined contact plan.

##Features

The **Simple-mpc** library provides:

* an interface to generate different locomotion gaits in a MPC-like fashion
* Python bindings to enable fast prototyping
* three different kinds of locomotion dynamics (centroidal, kinodynamics and full dynamics)

## Installation

### Build from source

```bash
git clone git@github.com:edantec/simple-mpc.git --recursive
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/ && cd build/
cmake --build . -jNCPUS
```

#### Dependencies

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* OpenMP
* [hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* [example-robot-data](https://github.com/Gepetto/example-robot-data)
* [ndcurves](https://github.com/loco-3d/ndcurves)
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.9.0 (Python bindings)
* (optional) [bullet](https://github.com/bulletphysics/bullet3) (Simulation examples)
