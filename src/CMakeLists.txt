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
option(SPDL_BUILD_TESTS "Build libspdl C++ unit tests" OFF)

if (SPDL_BUILD_TESTS)
  enable_testing()
endif()

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

if (SPDL_BUILD_TESTS)
  set(SPDL_TEST_FFMPEG_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH
    "Prefix containing full FFmpeg headers and libraries for native tests")
  set(FFMPEG_ROOT "${SPDL_TEST_FFMPEG_ROOT}")
  add_subdirectory(third_party/ffmpeg/single)

  # Name the core target for the actual FFmpeg ABI, never a link-time skeleton.
  set(codec_version_header "${FFMPEG_ROOT}/include/libavcodec/version_major.h")
  if (NOT EXISTS "${codec_version_header}")
    set(codec_version_header "${FFMPEG_ROOT}/include/libavcodec/version.h")
  endif()
  file(STRINGS "${codec_version_header}" codec_version_line
    REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+[0-9]+")
  string(REGEX REPLACE ".*LIBAVCODEC_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
    codec_major "${codec_version_line}")
  if (NOT codec_major MATCHES "^[0-9]+$")
    message(FATAL_ERROR "Could not determine the native FFmpeg codec ABI.")
  endif()
  math(EXPR native_ffmpeg_version "${codec_major} - 54")
  if (NOT native_ffmpeg_version MATCHES "^[4-9]$")
    message(FATAL_ERROR "Unsupported native FFmpeg major: ${native_ffmpeg_version}")
  endif()
  if (NOT DEFINED SPDL_USE_FFMPEG_VERSION OR SPDL_USE_FFMPEG_VERSION STREQUAL "all")
    set(SPDL_USE_FFMPEG_VERSION "${native_ffmpeg_version}")
  elseif (NOT SPDL_USE_FFMPEG_VERSION STREQUAL native_ffmpeg_version)
    message(FATAL_ERROR
      "Requested FFmpeg ${SPDL_USE_FFMPEG_VERSION}, but ${FFMPEG_ROOT} has FFmpeg ${native_ffmpeg_version} headers.")
  endif()
  add_library("ffmpeg${SPDL_USE_FFMPEG_VERSION}" ALIAS ffmpeg)
else()
  # Wheel builds continue to use the existing multi-version link-time skeletons.
  add_subdirectory(third_party/ffmpeg/multi)
endif()
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
