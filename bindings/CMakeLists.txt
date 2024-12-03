include("${JRL_CMAKE_MODULES}/python.cmake")

set(${PY_NAME}_SOURCES main.cpp expose-robot-handler.cpp expose-problem.cpp
                       expose-mpc.cpp expose-lowlevel.cpp problems.hpp)

python3_add_library(${PY_NAME}_pywrap MODULE WITH_SOABI ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
set_target_properties(
  ${PY_NAME}_pywrap
  PROPERTIES INSTALL_RPATH "\$ORIGIN/../../.."
             LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
             PREFIX ""
             SUFFIX ${PYTHON_EXT_SUFFIX})
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py)

foreach(python ${${PY_NAME}_PYTHON})
  python_install_on_site(${PY_NAME} ${python})
endforeach()
