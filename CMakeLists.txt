cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

option(SPDL_USE_CUDA "Enable CUDA support" OFF)
option(SPDL_USE_NVCODEC "Enable NVCODEC support" OFF)
option(SPDL_USE_NVJPEG "Enable nvJPEG support" OFF)
option(SPDL_DEBUG_REFCOUNT "Enable debug print for reference counting of AVFrame objects." OFF)
option(SPDL_BUILD_STUB "Build Python binding stub file" OFF)
option(SPDL_BUILD_PYTHON_BINDING "Build Python binding" ON)

if (SPDL_USE_NVCODEC OR SPDL_USE_NVJPEG)
  if (NOT SPDL_USE_CUDA)
    unset(SPDL_USE_CUDA CACHE)
    set(SPDL_USE_CUDA ON)
  endif()
endif()

###############################################################################
# Configurations
###############################################################################
include(./cmake/spdl_common_config.cmake)
include(./cmake/check_cxx_coroutine.cmake)
if (SPDL_USE_CUDA)
  if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  include(./cmake/spdl_cuda_config.cmake)
endif()

###############################################################################
# Build folly, libspdl and binding
###############################################################################
message(STATUS "########################################")
message(STATUS "# Looking for folly dependencies")
message(STATUS "########################################")
set(folly_deps double-conversion Libevent fmt gflags glog)
foreach (folly_dep IN LISTS folly_deps)
  find_package("${folly_dep}" REQUIRED)
endforeach()
find_package(Boost REQUIRED COMPONENTS algorithm intrusive context preprocessor variant iterator)

set(folly_deps double-conversion::double-conversion fmt::fmt gflags glog::glog libevent::core)
foreach(folly_dep IN LISTS folly_deps)
get_target_property(location "${folly_dep}" IMPORTED_LOCATION_RELEASE)
message(STATUS "Found ${folly_dep}: ${location}")
endforeach()

add_subdirectory(third_party/folly)
add_subdirectory(third_party/ffmpeg/multi)
if (SPDL_USE_TRACING)
  add_subdirectory(third_party/perfetto)
endif()
if (SPDL_USE_NVCODEC)
  add_subdirectory(third_party/nvcodec)
endif()

if (NOT DEFINED SPDL_PYTHON_BINDING_INSTALL_PREFIX)
  set(SPDL_PYTHON_BINDING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif ()

add_subdirectory(src)

if (SPDL_BUILD_PYTHON_BINDING)
  add_subdirectory(third_party/nanobind)
  add_subdirectory(src/python)
endif()

if (SPDL_BUILD_SAMPLES)
  add_subdirectory(src/prototypes)
endif (SPDL_BUILD_SAMPLES)

message(STATUS "########################################")
message(STATUS "# Configuration done!")
message(STATUS "########################################")
