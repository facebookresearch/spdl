cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(folly-deps)

###############################################################################
# Global config
###############################################################################
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

###############################################################################
# CCache
###############################################################################
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
  message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
endif()

###############################################################################
# FetchContent config
###############################################################################
include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# somehow `cmake_policy(SET .. NEW)` does not work for those...
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

###############################################################################
# Build folly dependencies.
###############################################################################
# glog has some private header file, which name conflicts with clang's tooling,
# so we need to install glog somewhere once.

# glog exports package to user registery (~/.glog) with build path and this
# causes issue when compiling folly.
# https://github.com/google/glog/issues/705
# the following disables the export.
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY true)

add_subdirectory(double-conversion)
add_subdirectory(fmt)
add_subdirectory(gflags)
add_subdirectory(glog)
add_subdirectory(libevent)
add_subdirectory(boost)
