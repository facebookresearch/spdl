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
  cuda/transfer.cpp
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
    cuda/utils.cpp
    cuda/buffer.cpp
    cuda/storage.cpp
    )
endif()

if (SPDL_USE_NVCODEC)
  list(APPEND defs SPDL_USE_NVCODEC)
  list(APPEND srcs
    cuda/nvdec/detail/buffer.cpp
    cuda/nvdec/detail/converter.cpp
    cuda/nvdec/detail/color_conversion.cu
    cuda/nvdec/detail/decoder.cpp
    cuda/nvdec/detail/utils.cpp
    cuda/nvdec/detail/wrapper.cpp
    )
  list(APPEND deps nvcuvid-stub)
endif()

if (SPDL_USE_NVJPEG)
  list(APPEND defs SPDL_USE_NVJPEG)
  list(APPEND srcs
    cuda/nvjpeg/detail/utils.cpp
    cuda/nvjpeg/detail/decoding.cpp
  )
  if (SPDL_USE_NPPI)
    list(APPEND defs SPDL_USE_NPPI)
    list(APPEND srcs
      cuda/npp/detail/resize.cpp
      cuda/npp/detail/utils.cpp
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
