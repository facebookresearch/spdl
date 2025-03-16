/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>
#include <libspdl/cuda/decoding.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVCODEC
#include "libspdl/cuda/nvdec/detail/decoder.h"
#endif

#include <fmt/core.h>

namespace spdl::core {

#ifdef SPDL_USE_NVCODEC
namespace {
void validate_nvdec_params(
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height) {
  if (cuda_device_index < 0) {
    SPDL_FAIL(fmt::format(
        "cuda_device_index must be non-negative. Found: {}",
        cuda_device_index));
  }
  if (crop.left < 0) {
    SPDL_FAIL(
        fmt::format("crop.left must be non-negative. Found: {}", crop.left));
  }
  if (crop.top < 0) {
    SPDL_FAIL(
        fmt::format("crop.top must be non-negative. Found: {}", crop.top));
  }
  if (crop.right < 0) {
    SPDL_FAIL(
        fmt::format("crop.right must be non-negative. Found: {}", crop.right));
  }
  if (crop.bottom < 0) {
    SPDL_FAIL(fmt::format(
        "crop.bottom must be non-negative. Found: {}", crop.bottom));
  }
  if (width > 0 && width % 2) {
    SPDL_FAIL(fmt::format("width must be positive and even. Found: {}", width));
  }
  if (height > 0 && height % 2) {
    SPDL_FAIL(
        fmt::format("height must be positive and even. Found: {}", height));
  }
}
} // namespace
#endif

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

#ifndef SPDL_USE_NVCODEC
NvDecDecoder::NvDecDecoder() : decoder(nullptr) {
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
}
#else
NvDecDecoder::NvDecDecoder() : decoder(new detail::NvDecDecoderInternal()) {}
#endif

NvDecDecoder::~NvDecDecoder() {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (decoder) {
    delete decoder;
  }
#endif
}

void NvDecDecoder::reset() {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  decoder->reset();
#endif
}

void NvDecDecoder::set_init_flag() {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  init = true;
#endif
}

CUDABufferPtr NvDecDecoder::decode(
    VideoPacketsPtr&& packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt) {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  validate_nvdec_params(cuda_config.device_index, crop, width, height);

  packets = apply_bsf(std::move(packets));

  if (init) {
    decoder->init(
        cuda_config.device_index,
        packets->codecpar->codec_id,
        packets->time_base,
        packets->timestamp,
        crop,
        width,
        height,
        pix_fmt);
    init = false;
  }
  return decoder->decode(
      std::move(packets), cuda_config, crop, width, height, pix_fmt);
#endif
}

} // namespace spdl::core
