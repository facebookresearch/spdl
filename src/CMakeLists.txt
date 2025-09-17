cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(spdl)

option(SPDL_USE_CUDA "Enable CUDA support" OFF)
option(SPDL_USE_NVCODEC "Enable NVCODEC support" OFF)
option(SPDL_USE_NVJPEG "Enable nvJPEG support" OFF)
option(SPDL_USE_NPPI "Enable NVIDIA 2D Image And Signal Performance Primitives support" OFF)
option(SPDL_LINK_STATIC_NVJPEG "Link nvJPEG and NPPI statically." OFF)

option(SPDL_DEBUG_REFCOUNT "Enable debug print for reference counting of AVFrame objects." OFF)
option(SPDL_IS_GIL_ENABLED "Whether the target Python has the GIL enabled" ON)
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
include(./_cmake/spdl_common_config.cmake)
include(./_cmake/check_cxx_coroutine.cmake)
if (SPDL_USE_CUDA)
  if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  include(./_cmake/spdl_cuda_config.cmake)
endif()

###############################################################################
# Build libspdl and binding
###############################################################################
message(STATUS "########################################")
message(STATUS "# Looking for third party dependencies")
message(STATUS "########################################")
find_package(fmt REQUIRED NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
find_package(glog REQUIRED NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
find_package(libdeflate REQUIRED NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
find_package(ZLIB REQUIRED NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)

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

message(STATUS "SPDL_CXX_CPU_COMPILE_FLAGS=${SPDL_CXX_CPU_COMPILE_FLAGS}")

add_subdirectory(libspdl)

if (SPDL_BUILD_PYTHON_BINDING)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  add_subdirectory(third_party/nanobind)
  add_subdirectory(spdl/io/lib)
endif()

message(STATUS "########################################")
message(STATUS "# Configuration done!")
message(STATUS "########################################")
