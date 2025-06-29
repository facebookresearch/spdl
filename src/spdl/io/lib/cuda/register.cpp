/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SPDL_CUDA_EXT_NAME
#error SPDL_CUDA_EXT_NAME must be defined.
#endif

#include "register_spdl_cuda_extensions.h"

#include <nanobind/nanobind.h>

namespace {
NB_MODULE(SPDL_CUDA_EXT_NAME, m) {
  spdl::cuda::register_types(m);
  spdl::cuda::register_buffers(m);
  spdl::cuda::register_decoding_nvdec(m);
  spdl::cuda::register_decoding_nvjpeg(m);
  spdl::cuda::register_storage(m);
  spdl::cuda::register_transfer(m);
  spdl::cuda::register_utils(m);
  spdl::cuda::register_color_conversion(m);
}
} // namespace
