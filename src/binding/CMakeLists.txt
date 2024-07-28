# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring SPDL Python binding")
message(STATUS "########################################")

set(src
  bind.cpp
  core/adaptors.cpp
  core/frames.cpp
  core/buffer.cpp
  core/packets.cpp
  core/types.cpp
  core/tracing.cpp
  core/utils.cpp
  core/demuxing.cpp
  core/decoding.cpp
  core/conversion.cpp
  core/transfer.cpp
  core/encoding.cpp
)

# This adds the library directory to search path
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")
  message(STATUS "Building ${name}")

  set(defs "-DSPDL_FFMPEG_EXT_NAME=${name}")
  if (SPDL_USE_CUDA)
    list(APPEND defs SPDL_USE_CUDA)
  endif()
  if (SPDL_USE_NVCODEC)
    list(APPEND defs SPDL_USE_NVCODEC)
  endif ()

  nanobind_add_module(${name} ${src})
  target_compile_definitions(${name} PRIVATE "${defs}")
  target_link_libraries(${name} PRIVATE "spdl_ffmpeg${ffmpeg_version}")
  target_include_directories(${name} PRIVATE "${Python_INCLUDE_DIR}")
  target_include_directories(nanobind-static PRIVATE "${Python_INCLUDE_DIR}")

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
