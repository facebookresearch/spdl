/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>

#ifndef SPDL_CUDA_EXT_NAME
#error SPDL_CUDA_EXT_NAME must be defined.
#endif

namespace nb = nanobind;

namespace spdl::cuda {
void register_types(nb::module_&);
void register_buffers(nb::module_&);
void register_decoding(nb::module_&);
void register_storage(nb::module_&);
void register_transfer(nb::module_&);
void register_encoding(nb::module_&);
void register_utils(nb::module_&);
void register_color_conversion(nb::module_&);
} // namespace spdl::cuda

namespace {
NB_MODULE(SPDL_CUDA_EXT_NAME, m) {
  spdl::cuda::register_types(m);
  spdl::cuda::register_buffers(m);
  spdl::cuda::register_decoding(m);
  spdl::cuda::register_storage(m);
  spdl::cuda::register_transfer(m);
  spdl::cuda::register_encoding(m);
  spdl::cuda::register_utils(m);
  spdl::cuda::register_color_conversion(m);
}
} // namespace
