# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring SPDL")
message(STATUS "########################################")

# This adds the library directory to search path
if (SPDL_BUILD_PYTHON_BINDING)
  if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
  elseif (UNIX)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
  endif ()
endif()

set(defs)
set(deps fmt::fmt glog::glog)
set(srcs
  core/detail/ffmpeg/conversion.cpp
  core/detail/ffmpeg/ctx_utils.cpp
  core/detail/ffmpeg/bsf.cpp
  core/detail/ffmpeg/decoder.cpp
  core/detail/ffmpeg/demuxer.cpp
  core/detail/ffmpeg/demuxing.cpp
  core/detail/ffmpeg/encoding.cpp
  core/detail/ffmpeg/filter_graph.cpp
  core/detail/ffmpeg/wrappers.cpp
  core/detail/logging.cpp
  core/buffer/cpu.cpp
  core/frames/ffmpeg.cpp
  core/adaptor/adaptor.cpp
  core/adaptor/bytes.cpp
  core/storage/cpu.cpp
  core/transfer.cpp
  core/decoder.cpp
  core/decoding.cpp
  core/demuxing.cpp
  core/encoding.cpp
  core/packets.cpp
  core/utils.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
  list(APPEND srcs
    core/detail/cuda.cpp
    core/buffer/cuda.cpp
    core/storage/cuda.cpp
    )
endif()

if (SPDL_USE_NVCODEC)
  list(APPEND defs SPDL_USE_NVCODEC)
  list(APPEND deps nvcuvid-stub)
endif()

if (SPDL_USE_NVJPEG)
  list(APPEND defs SPDL_USE_NVJPEG)
  list(APPEND srcs
    core/detail/nvjpeg/utils.cpp
    core/detail/nvjpeg/decoding.cpp
  )
  if (SPDL_USE_NPPI)
    list(APPEND defs SPDL_USE_NPPI)
    list(APPEND srcs
      core/detail/npp/resize.cpp
      core/detail/npp/utils.cpp
    )
  endif()

  if (SPDL_LINK_STATIC_NVJPEG)
    list(APPEND deps CUDA::nvjpeg_static)
    if (SPDL_USE_NPPI)
      list(APPEND deps CUDA::nppig_static)
    endif()
  else()
    list(APPEND deps CUDA::nvjpeg)
    if (SPDL_USE_NPPI)
      list(APPEND deps CUDA::nppig)
    endif()
  endif()
endif()

if(SPDL_USE_TRACING)
  list(APPEND deps perfetto)
  list(APPEND defs SPDL_USE_TRACING)
  list(APPEND srcs core/detail/tracing.cpp)
endif()

if (SPDL_DEBUG_REFCOUNT)
  list(APPEND defs SPDL_DEBUG_REFCOUNT)
endif()

# install(
#   DIRECTORY libspdl
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#   FILES_MATCHING
#   PATTERN "*.h"
#   PATTERN "detail" EXCLUDE)

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  message(STATUS "Building ${name}")
  add_library("${name}" SHARED "${srcs}")
  target_compile_definitions("${name}" PRIVATE "${defs}")
  target_link_libraries("${name}" PUBLIC "${deps}" "ffmpeg${ffmpeg_version}")
  target_include_directories(
    "${name}"
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/.."
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

  install(
    TARGETS "${name}"
    EXPORT "${name}Targets"
    LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
    RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
    )

endfunction()

set(ffmpeg_versions 4 5 6 7)
if (SPDL_USE_FFMPEG_VERSION IN_LIST ffmpeg_versions)
add_libspdl("${SPDL_USE_FFMPEG_VERSION}")
else()
add_libspdl(4)
add_libspdl(5)
add_libspdl(6)
add_libspdl(7)
endif()

# -------
# NVDEC extension

if (SPDL_USE_NVCODEC)
  SET(srcs
    core/detail/nvdec/foo.cpp
    core/detail/nvdec/buffer.cpp
    core/detail/nvdec/converter.cpp
    core/detail/nvdec/color_conversion.cu
    core/detail/nvdec/decoder.cpp
    core/detail/nvdec/utils.cpp
    core/detail/nvdec/wrapper.cpp
    )


  function(add_libspdl_nvcodec ffmpeg_version)
    set(name "spdl_nvcodec_ffmpeg${ffmpeg_version}")
    message(STATUS "Building ${name}")
    add_library("${name}" SHARED "${srcs}")
    target_compile_definitions("${name}" PRIVATE "${defs}")
    target_link_libraries("${name}" PUBLIC nvcuvid-stub "spdl_ffmpeg${ffmpeg_version}")
    target_include_directories(
      "${name}"
      PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/.."
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
    
    install(
      TARGETS "${name}"
      EXPORT "${name}Targets"
      LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
      RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
      )
    
  endfunction()

  set(ffmpeg_versions 4 5 6 7)
  if (SPDL_USE_FFMPEG_VERSION IN_LIST ffmpeg_versions)
    add_libspdl_nvcodec("${SPDL_USE_FFMPEG_VERSION}")
  else()
    add_libspdl_nvcodec(4)
    add_libspdl_nvcodec(5)
    add_libspdl_nvcodec(6)
    add_libspdl_nvcodec(7)
  endif()

endif()
