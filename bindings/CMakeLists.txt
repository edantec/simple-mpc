include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/stubs.cmake")

add_project_dependency(eigenpy 2.7.10 REQUIRED)

set(
  ${PY_NAME}_SOURCES
  module.cpp
  expose-robot-handler.cpp
  expose-problem.cpp
  expose-mpc.cpp
  expose-lowlevel.cpp
  problems.hpp
)

python3_add_library(${PY_NAME}_pywrap MODULE WITH_SOABI ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
set_target_properties(
  ${PY_NAME}_pywrap
  PROPERTIES
    INSTALL_RPATH "\$ORIGIN/../../.."
    LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
)
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py)

foreach(python ${${PY_NAME}_PYTHON})
  python_install_on_site(${PY_NAME} ${python})
endforeach()

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} simple_mpc ${PYTHON_SITELIB}
                 ${PY_NAME}_pywrap
  )
endif()
