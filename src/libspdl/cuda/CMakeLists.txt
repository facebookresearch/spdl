# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring libspdl (cuda)")
message(STATUS "########################################")

# This adds the library directory to search path
if (SPDL_BUILD_PYTHON_BINDING)
  if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
  elseif (UNIX)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
  endif ()
endif()

set(deps
  fmt::fmt
  glog::glog
  CUDA::cudart
  CUDA::cuda_driver
)
set(srcs
  utils.cpp
  buffer.cpp
  storage.cpp
  transfer.cpp
)

if (SPDL_USE_NVCODEC)
  list(APPEND srcs
    nvdec/decoder.cpp
    nvdec/detail/buffer.cpp
    nvdec/detail/converter.cpp
    nvdec/detail/color_conversion.cu
    nvdec/detail/decoder.cpp
    nvdec/detail/utils.cpp
    nvdec/detail/wrapper.cpp
    )
  list(APPEND deps nvcuvid-stub)
endif()

if (SPDL_USE_NVJPEG)
  list(APPEND srcs
    nvjpeg/decoding.cpp
    nvjpeg/detail/utils.cpp
    nvjpeg/detail/decoding.cpp
  )
  if (SPDL_USE_NPPI)
    list(APPEND srcs
      npp/detail/resize.cpp
      npp/detail/utils.cpp
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
endif()

function(add_libspdl ffmpeg_version)
  set(name "spdl_cuda${ffmpeg_version}")
  message(STATUS "Building ${name}")
  add_library("${name}" SHARED "${srcs}")
  target_compile_definitions("${name}" PRIVATE "${SPDL_COMMON_DEFS}")
  target_link_libraries("${name}" PRIVATE "${deps}" "spdl_ffmpeg${ffmpeg_version}")
  target_include_directories(
    "${name}"
    PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../.."
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
