set(${PY_NAME}_SOURCES main.cpp expose-robot-handler.cpp expose-problem.cpp
                       expose-mpc.cpp expose-lowlevel.cpp problems.hpp)

add_library(${PY_NAME}_pywrap SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
target_include_directories(${PY_NAME}_pywrap SYSTEM
                           PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(
  ${PY_NAME}_pywrap
  PROPERTIES INSTALL_RPATH "\$ORIGIN/../../.."
             LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
             PREFIX ""
             SUFFIX ${PYTHON_EXT_SUFFIX})
target_link_boost_python(${PY_NAME}_pywrap PRIVATE)
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py)

foreach(python ${${PY_NAME}_PYTHON})
  python_install_on_site(${PY_NAME} ${python})
endforeach()
