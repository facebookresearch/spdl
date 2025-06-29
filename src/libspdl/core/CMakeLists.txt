# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring libspdl (core)")
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
  detail/ffmpeg/conversion.cpp
  detail/ffmpeg/ctx_utils.cpp
  detail/ffmpeg/bsf.cpp
  detail/ffmpeg/encoder.cpp
  detail/ffmpeg/decoder.cpp
  detail/ffmpeg/demuxer.cpp
  detail/ffmpeg/muxer.cpp
  detail/ffmpeg/filter_graph.cpp
  detail/ffmpeg/wrappers.cpp
  detail/logging.cpp
  adaptor/adaptor.cpp
  adaptor/bytes.cpp
  encoder.cpp
  bsf.cpp
  frames.cpp
  filter_graph.cpp
  buffer.cpp
  storage.cpp
  decoder.cpp
  demuxing.cpp
  codec.cpp
  muxer.cpp
  packets.cpp
  utils.cpp
  )

if(SPDL_USE_TRACING)
  list(APPEND deps perfetto)
  list(APPEND srcs detail/tracing.cpp)
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
  target_compile_definitions("${name}" PUBLIC "${SPDL_COMMON_DEFS}")
  target_compile_definitions("${name}" PRIVATE "${defs}")
  target_link_libraries("${name}" PRIVATE "${deps}" "ffmpeg${ffmpeg_version}")
  target_include_directories(
    "${name}"
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_compile_options("${name}" PRIVATE "${SPDL_CXX_CPU_COMPILE_FLAGS}")

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
