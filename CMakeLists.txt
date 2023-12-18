cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

option(USE_CUDA "Enable CUDA support" OFF)

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
# Build folly, libspdl and binding
###############################################################################
find_package(double-conversion REQUIRED)
find_package(Libevent REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS algorithm intrusive context preprocessor variant iterator)

add_subdirectory(third_party/folly)
add_subdirectory(third_party/ffmpeg/multi)
add_subdirectory(third_party/pybind11)
add_subdirectory(src/libspdl)
if (SPDL_BUILD_PYTHON_BINDING)
  add_subdirectory(src/pybind)
endif()

if (SPDL_BUILD_SAMPLES)
  add_subdirectory(src/prototypes)
endif (SPDL_BUILD_SAMPLES)
