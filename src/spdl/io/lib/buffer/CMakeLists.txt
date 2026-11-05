# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
Python_add_library(_buffer MODULE WITH_SOABI register.cpp)
target_compile_options(_buffer PRIVATE ${SPDL_CXX_CPU_COMPILE_FLAGS})

install(
  TARGETS _buffer
  LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
)
