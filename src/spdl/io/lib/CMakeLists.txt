# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This adds the library directory to search path
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

message(STATUS "########################################")
message(STATUS "# Configuring SPDL Python binding (core)")
message(STATUS "########################################")

set(src
  core/register.cpp
  core/buffer.cpp
  core/conversion.cpp
  core/decoding.cpp
  core/demuxing.cpp
  core/encoding.cpp
  core/frames.cpp
  core/packets.cpp
  core/storage.cpp
  core/tracing.cpp
  core/types.cpp
  core/utils.cpp
)

if (NOT SPDL_IS_GIL_ENABLED)
  set(ft FREE_THREADED)
endif()

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")
  message(STATUS "Building ${name}")

  set(defs "-DSPDL_FFMPEG_EXT_NAME=${name}")
  set(deps "spdl_ffmpeg${ffmpeg_version}")
  if (SPDL_USE_CUDA)
    list(APPEND defs SPDL_USE_CUDA)
    list(APPEND deps "spdl_cuda${ffmpeg_version}")
  endif()
  if (SPDL_USE_NVCODEC)
    list(APPEND defs SPDL_USE_NVCODEC)
  endif ()
  if (SPDL_HOLD_GIL)
    list(APPEND defs SPDL_HOLD_GIL)
  endif ()

  nanobind_add_module(${name} ${ft} ${src})
  target_compile_definitions(${name} PRIVATE "${defs}")
  target_link_libraries(${name} PRIVATE "${deps}")
  target_include_directories(${name} PRIVATE "${Python_INCLUDE_DIR}")

  install(
    TARGETS ${name}
    LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
    RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )

  if (SPDL_BUILD_STUB)
    if (NOT TARGET "_libspdl_stub")
      nanobind_add_stub(
        _libspdl_stub
        MODULE ${name}
        OUTPUT _libspdl.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${name}>
        DEPENDS ${name}
        )

      install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/_libspdl.pyi"
        DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
        )
    endif()
  endif()

endfunction()

set(ffmpeg_versions 4 5 6 7)
if (SPDL_USE_FFMPEG_VERSION IN_LIST ffmpeg_versions)
add_spdl_extension("${SPDL_USE_FFMPEG_VERSION}")
else()
add_spdl_extension(4)
add_spdl_extension(5)
add_spdl_extension(6)
add_spdl_extension(7)
endif()

message(STATUS "########################################")
message(STATUS "# Configuring SPDL Python binding (cuda)")
message(STATUS "########################################")

set(src
  cuda/register.cpp
  cuda/buffer.cpp
  cuda/decoding.cpp
  cuda/encoding.cpp
  cuda/storage.cpp
  cuda/transfer.cpp
  cuda/types.cpp
)

if (NOT SPDL_IS_GIL_ENABLED)
  set(ft FREE_THREADED)
endif()

function(add_spdl_cuda_extension ffmpeg_version)
  set(name "_spdl_cuda${ffmpeg_version}")
  message(STATUS "Building ${name}")

  set(defs "-DSPDL_CUDA_EXT_NAME=${name}")
  set(deps "spdl_ffmpeg${ffmpeg_version}")
  if (SPDL_USE_CUDA)
    list(APPEND defs SPDL_USE_CUDA)
    list(APPEND deps "spdl_cuda${ffmpeg_version}")
  endif()
  if (SPDL_USE_NVCODEC)
    list(APPEND defs SPDL_USE_NVCODEC)
  endif ()
  if (SPDL_HOLD_GIL)
    list(APPEND defs SPDL_HOLD_GIL)
  endif ()

  nanobind_add_module(${name} ${ft} ${src})
  target_compile_definitions(${name} PRIVATE "${defs}")
  target_link_libraries(${name} PRIVATE "${deps}")
  target_include_directories(${name} PRIVATE "${Python_INCLUDE_DIR}")

  install(
    TARGETS ${name}
    LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
    RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )

endfunction()

set(ffmpeg_versions 4 5 6 7)
if (SPDL_USE_FFMPEG_VERSION IN_LIST ffmpeg_versions)
add_spdl_cuda_extension("${SPDL_USE_FFMPEG_VERSION}")
else()
add_spdl_cuda_extension(4)
add_spdl_cuda_extension(5)
add_spdl_cuda_extension(6)
add_spdl_cuda_extension(7)
endif()

###############################################################################
# zip extension
###############################################################################

set(name _zip)
message(STATUS "Building ${name}")

set(srcs zip.cpp)
set(deps libzip::zip fmt::fmt glog::glog)
nanobind_add_module("${name}" "${srcs}")
target_link_libraries("${name}" PRIVATE "${deps}")
target_include_directories("${name}" PRIVATE "${Python_INCLUDE_DIR}")
target_include_directories(nanobind-static PRIVATE "${Python_INCLUDE_DIR}")

install(
  TARGETS ${name}
  LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )
