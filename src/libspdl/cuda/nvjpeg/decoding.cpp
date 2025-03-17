/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>
#include <libspdl/cuda/nvjpeg/decoding.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

namespace spdl::core {

#ifdef SPDL_USE_NVJPEG
namespace detail {
CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);
CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);
} // namespace detail
#endif

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
  SPDL_FAIL("SPDL is not compiled with NVJPEG support.");
#else
  return detail::decode_image_nvjpeg(
      data, cuda_config, scale_width, scale_height, pix_fmt);
#endif
}

CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
  SPDL_FAIL("SPDL is not compiled with NVJPEG support.");
#else
  return detail::decode_image_nvjpeg(
      data, cuda_config, scale_width, scale_height, pix_fmt);
#endif
}

} // namespace spdl::core
