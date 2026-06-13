# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(SPDL_COMMON_DEFS)
if (SPDL_USE_TRACING)
  list(APPEND SPDL_COMMON_DEFS SPDL_USE_TRACING)
endif ()
if (SPDL_USE_CUDA)
  list(APPEND SPDL_COMMON_DEFS SPDL_USE_CUDA)
endif ()
if (SPDL_USE_NVCODEC)
  list(APPEND SPDL_COMMON_DEFS SPDL_USE_NVCODEC)
endif ()
if (SPDL_USE_NVJPEG)
  list(APPEND SPDL_COMMON_DEFS SPDL_USE_NVJPEG)
endif ()
if (SPDL_USE_NPPI)
  list(APPEND SPDL_COMMON_DEFS SPDL_USE_NPPI)
endif ()

if (MSVC)
   # Temporary.
   # TODO: Set the visibility of public functions.
   set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(MSVC)

add_subdirectory(core)
if (SPDL_USE_CUDA)
add_subdirectory(cuda)
endif()

# Enable testing
if (SPDL_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
