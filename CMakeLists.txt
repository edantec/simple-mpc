cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAME simple-mpc)
set(PROJECT_DESCRIPTION "MPC schemes for locomotion")
set(PROJECT_URL https://github.com/edantec/${PROJECT_NAME})

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
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
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

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_compile_options(-Wno-error)

option(ENABLE_TRACY_PROFILING "Enable Tracy profiler" OFF)
option(DOWNLOAD_TRACY "Use FetchContent to install Tracy." ON)
if(DOWNLOAD_TRACY)
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/Simple-Robotics/tracy.git
    GIT_TAG patches
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_GetProperties(tracy)
  if(NOT tracy_POPULATED)
    FetchContent_Populate(tracy)
    set(TRACY_STATIC
        ON
        CACHE INTERNAL "")
    set(TRACY_ENABLE
        ${ENABLE_TRACY_PROFILING}
        CACHE INTERNAL "")
    add_subdirectory(${tracy_SOURCE_DIR} ${tracy_BINARY_DIR} EXCLUDE_FROM_ALL)
    # Extract the target include directories, set as system
    get_target_property(tracy_INCLUDE_DIR TracyClient
                        INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
      TracyClient
      PROPERTIES POSITION_INDEPENDENT_CODE True
                 INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${tracy_INCLUDE_DIR}")
  endif()
else()
  # assume it is installed somewhere
  find_package(Tracy REQUIRED)
  set_target_properties(Tracy::TracyClient PROPERTIES POSITION_INDEPENDENT_CODE
                                                      True)
endif()

find_package(aligator REQUIRED)
find_package(Threads REQUIRED)
add_project_dependency(OpenMP REQUIRED)
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(proxsuite-nlp 0.6.1 REQUIRED PKG_CONFIG_REQUIRES
                       "proxsuite-nlp >= 0.6.1")
add_project_dependency(aligator REQUIRED)

# Main Library
set(${PROJECT_NAME}_HEADERS include/${PROJECT_NAME}/fwd.hpp
                            include/${PROJECT_NAME}/fulldynamics.hpp)

set(${PROJECT_NAME}_SOURCES src/fulldynamics.cpp)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                   ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC pinocchio::pinocchio proxsuite-nlp::proxsuite-nlp
                         aligator::aligator)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
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

# Installation
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  DESTINATION lib)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

if(DOWNLOAD_TRACY)
  install(
    TARGETS TracyClient
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/aligator
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/aligator)
  message(
    STATUS
      "Installing TracyClient target to ${CMAKE_INSTALL_FULL_LIBDIR}/aligator")
endif()
