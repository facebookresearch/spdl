# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring SPDL Python binding (archive)")
message(STATUS "########################################")

set(name _archive)
message(STATUS "Building ${name}")

set(srcs register.cpp zip_impl.cpp)
set(deps ZLIB::ZLIB fmt::fmt glog::glog)
nanobind_add_module("${name}" "${srcs}")
target_link_libraries("${name}" PRIVATE "${deps}")
target_include_directories("${name}" PRIVATE "${Python_INCLUDE_DIR}")
target_include_directories(nanobind-static PRIVATE "${Python_INCLUDE_DIR}")

install(
  TARGETS ${name}
  LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )
