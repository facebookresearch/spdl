cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(folly-subset)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# This adds the library directory to search path
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

option(SPDL_BUILD_SAMPLES "Build sample codes." OFF)

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
# Third party libraries
###############################################################################
include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# somehow `cmake_policy(SET .. NEW)` does not work for those...
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/ffmpeg/multi)
add_subdirectory(third_party/double-conversion)
add_subdirectory(third_party/fmt)
# add_subdirectory(third_party/gflags)
add_subdirectory(third_party/glog)
add_subdirectory(third_party/libevent)
add_subdirectory(third_party/boost)
add_subdirectory(third_party/folly)


###############################################################################
# SPDL
###############################################################################
add_subdirectory(src/libspdl)
add_subdirectory(src/pybind)

###############################################################################
# Prototype / test code
###############################################################################
if (SPDL_BUILD_SAMPLES)
add_subdirectory(src/prototypes)
endif (SPDL_BUILD_SAMPLES)
