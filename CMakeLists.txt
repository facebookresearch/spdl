cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

include(./cmake/spdl_common_config.cmake)
option(SPDL_USE_CUDA "Enable CUDA support" OFF)
option(SPDL_DEBUG_REFCOUNT "Enable debug print for reference counting of AVFrame objects." OFF)

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
