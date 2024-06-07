cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(first-stage-deps)

include(../cmake/spdl_common_config.cmake)

# Make all the symbols from third party libraries private
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

###############################################################################
# Build third party dependencies.
###############################################################################
# glog has some private header file, which name conflicts with clang's tooling,
# so we need to install glog somewhere once.

# glog exports package to user registery (~/.glog) with build path and this
# causes issue when compiling SPDL.
# https://github.com/google/glog/issues/705
# the following disables the export.
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY true)

add_subdirectory(fmt)
add_subdirectory(gflags)
add_subdirectory(glog)
