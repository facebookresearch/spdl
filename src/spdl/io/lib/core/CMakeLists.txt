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
  register.cpp
  bsf.cpp
  buffer.cpp
  conversion.cpp
  decoding.cpp
  demuxing.cpp
  encoding.cpp
  frames.cpp
  filtering.cpp
  packets.cpp
  storage.cpp
  tracing.cpp
  types.cpp
  utils.cpp
)

set(nb_options
  STABLE_ABI
  NB_STATIC
  LTO
  NB_SUPPRESS_WARNINGS
  )

if (NOT SPDL_IS_GIL_ENABLED)
  list(APPEND nb_options FREE_THREADED)
endif()

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")
  message(STATUS "Building ${name}")

  set(defs "-DSPDL_FFMPEG_EXT_NAME=${name}")
  set(deps fmt::fmt glog::glog "spdl_ffmpeg${ffmpeg_version}")

  nanobind_add_module(${name} ${nb_options} ${src})
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
