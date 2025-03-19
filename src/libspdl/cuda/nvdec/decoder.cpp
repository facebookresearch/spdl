/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/nvdec/decoder.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVCODEC
#include "libspdl/cuda/nvdec/detail/decoder.h"
#include "libspdl/cuda/nvdec/detail/decoder_core.h"
#endif

#include <fmt/core.h>

namespace spdl::cuda {

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

NvDecDecoder::NvDecDecoder() : decoder(new detail::NvDecDecoderInternal()) {}

NvDecDecoder::~NvDecDecoder() {
  if (decoder) {
    delete decoder;
  }
}

void NvDecDecoder::reset() {
  decoder->reset();
}

void NvDecDecoder::set_init_flag() {
  init = true;
}

CUDABufferPtr NvDecDecoder::decode(
    spdl::core::VideoPacketsPtr&& packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool flush) {
  validate_nvdec_params(cuda_config.device_index, crop, width, height);

  if (init) {
    decoder->init(
        cuda_config.device_index,
        packets->get_codec().get_codec_id(),
        packets->time_base,
        packets->timestamp,
        crop,
        width,
        height,
        pix_fmt);
    init = false;
  }
  return decoder->decode(
      std::move(packets), cuda_config, crop, width, height, pix_fmt, flush);
}

////////////////////////////////////////////////////////////////////////////////
// NvDecoder2
////////////////////////////////////////////////////////////////////////////////

NvDecDecoder2::NvDecDecoder2() : core(new detail::NvDecDecoderCore2()) {}

NvDecDecoder2::~NvDecDecoder2() {
  if (core) {
    delete core;
  }
}

void NvDecDecoder2::reset() {
  core->reset();
}

void NvDecDecoder2::init(
    // device config
    const CUDAConfig& cuda_config,
    // Source codec information
    const spdl::core::VideoCodec& codec,
    // Post-decoding processing params
    CropArea crop,
    int target_width,
    int target_height) {
  core->init(cuda_config, codec, crop, target_width, target_height);
}

std::vector<CUDABuffer> NvDecDecoder2::decode(
    spdl::core::VideoPacketsPtr packets) {
  std::vector<CUDABuffer> ret;
  core->decode_packets(packets.get(), &ret);
  return ret;
}

std::vector<CUDABuffer> NvDecDecoder2::flush() {
  std::vector<CUDABuffer> ret;
  core->flush(&ret);
  return ret;
}

#endif
} // namespace spdl::cuda
