cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAME simple-mpc)
set(PROJECT_DESCRIPTION "MPC schemes for locomotion")
set(PROJECT_URL https://github.com/edantec/${PROJECT_NAME})
set(PROJECT_CUSTOM_HEADER_DIR "simple-mpc")

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)

# Project configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

include(FetchContent)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake to >= 3.14 to allow automatic fetching\n")
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_compile_options(-Wno-error)

find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)
find_package(Threads REQUIRED)
find_package(example-robot-data 4.0.9 REQUIRED)
find_package(Tracy REQUIRED)
set_target_properties(Tracy::TracyClient PROPERTIES POSITION_INDEPENDENT_CODE
                                                    True)
add_project_dependency(OpenMP REQUIRED)
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(proxsuite-nlp 0.6.1 REQUIRED PKG_CONFIG_REQUIRES
                       "proxsuite-nlp >= 0.6.1")
add_project_dependency(aligator REQUIRED)

# Project dependencies
if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 2.7.10 REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

# Main Library
file(GLOB mpc_SOURCE CONFIGURE_DEPENDS src/*.cpp)
file(GLOB mpc_HEADER CONFIGURE_DEPENDS include/${PROJECT_NAME}/*.hpp)

add_library(${PROJECT_NAME} SHARED ${mpc_HEADER} ${mpc_SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC pinocchio::pinocchio proxsuite-nlp::proxsuite-nlp aligator::aligator
         example-robot-data::example-robot-data)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${mpc_HEADER}")
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX
                                                 INSTALL_RPATH "\$ORIGIN")

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Unit tests
if(BUILD_TESTING)
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
  add_subdirectory(tests)
endif()

# Benchmark
if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

# Installation
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
