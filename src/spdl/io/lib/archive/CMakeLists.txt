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

set(srcs register.cpp zip_impl.cpp numpy_support.cpp)
set(deps ZLIB::ZLIBSTATIC fmt::fmt glog::glog libdeflate::libdeflate_static)
set(nb_options
  STABLE_ABI
  NB_STATIC
  LTO
  NB_SUPPRESS_WARNINGS
  )

if (NOT SPDL_IS_GIL_ENABLED)
  list(APPEND nb_options FREE_THREADED)
endif()

nanobind_add_module(${name} ${nb_options} ${srcs})
target_link_libraries(${name} PRIVATE "${deps}")
target_include_directories(${name} PRIVATE "${Python_INCLUDE_DIR}")
target_compile_options(${name} PRIVATE "${SPDL_CXX_CPU_COMPILE_FLAGS}")

install(
  TARGETS ${name}
  LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )
