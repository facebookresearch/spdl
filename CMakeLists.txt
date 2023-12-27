cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

include(./cmake/spdl_common_config.cmake)
option(SPDL_USE_CUDA "Enable CUDA support" OFF)
option(SPDL_DEBUG_REFCOUNT "Enable debug print for reference counting of AVFrame objects." OFF)

if (SPDL_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED COMPONENT cuda_driver cudart)
endif()

###############################################################################
# Build folly, libspdl and binding
###############################################################################
message(STATUS "########################################")
message(STATUS "# Looking for folly dependencies")
message(STATUS "########################################")
set(folly_deps double-conversion Libevent fmt gflags glog)
foreach (folly_dep IN LISTS folly_deps)
  message(STATUS "Looking for ${folly_dep}")
  find_package("${folly_dep}" REQUIRED)
endforeach()
message(STATUS "Looking for Boost")
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
