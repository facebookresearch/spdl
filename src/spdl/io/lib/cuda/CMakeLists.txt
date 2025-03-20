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
message(STATUS "# Configuring SPDL Python binding (cuda)")
message(STATUS "########################################")

set(src
  register.cpp
  buffer.cpp
  decoding.cpp
  encoding.cpp
  storage.cpp
  transfer.cpp
  types.cpp
  utils.cpp
  color_conversion.cpp
)

if (NOT SPDL_IS_GIL_ENABLED)
  set(ft FREE_THREADED)
endif()

function(add_spdl_cuda_extension ffmpeg_version)
  set(name "_spdl_cuda${ffmpeg_version}")
  message(STATUS "Building ${name}")

  set(defs "-DSPDL_CUDA_EXT_NAME=${name}")
  set(deps fmt::fmt glog::glog "spdl_ffmpeg${ffmpeg_version}")
  if (SPDL_USE_CUDA)
    list(APPEND deps "spdl_cuda${ffmpeg_version}")
  endif()

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
