cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

option(SPDL_USE_CUDA "Enable CUDA support" OFF)
option(SPDL_DEBUG_REFCOUNT "Enable debug print for reference counting of AVFrame objects." OFF)

if (SPDL_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED COMPONENT cuda_driver cudart)
endif()

###############################################################################
# Condigurations
###############################################################################
include(./cmake/spdl_common_config.cmake)
include(./cmake/check_cxx_coroutine.cmake)

###############################################################################
# Build folly, libspdl and binding
###############################################################################
message(STATUS "########################################")
message(STATUS "# Looking for folly dependencies")
message(STATUS "########################################")
set(folly_deps double-conversion fmt gflags glog)
foreach (folly_dep IN LISTS folly_deps)
  find_package("${folly_dep}" REQUIRED)
endforeach()
find_package(Boost REQUIRED COMPONENTS algorithm intrusive context preprocessor variant iterator)
find_package(Libevent REQUIRED COMPONENTS core)

set(folly_deps double-conversion::double-conversion fmt::fmt gflags glog::glog libevent::core)
foreach(folly_dep IN LISTS folly_deps)
get_target_property(location "${folly_dep}" IMPORTED_LOCATION_RELEASE)
message(STATUS "Found ${folly_dep}: ${location}")
endforeach()

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
