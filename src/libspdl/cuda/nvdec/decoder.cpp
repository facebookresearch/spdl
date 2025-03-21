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

#include "libspdl/cuda/nvdec/detail/decoder.h"

#include <fmt/core.h>

namespace spdl::cuda {

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

NvDecDecoder::NvDecDecoder() : core(new detail::NvDecDecoderCore()) {}

NvDecDecoder::~NvDecDecoder() {
  if (core) {
    delete core;
  }
}

void NvDecDecoder::reset() {
  core->reset();
}

void NvDecDecoder::init(
    // device config
    const CUDAConfig& cuda_config,
    // Source codec information
    const spdl::core::VideoCodec& codec,
    // Post-decoding processing params
    CropArea crop,
    int width,
    int height) {
  validate_nvdec_params(cuda_config.device_index, crop, width, height);
  core->init(cuda_config, codec, crop, width, height);
}

std::vector<CUDABuffer> NvDecDecoder::decode(
    spdl::core::VideoPacketsPtr packets) {
  std::vector<CUDABuffer> ret;
  core->decode_packets(packets.get(), &ret);
  return ret;
}

std::vector<CUDABuffer> NvDecDecoder::flush() {
  std::vector<CUDABuffer> ret;
  core->flush(&ret);
  return ret;
}
} // namespace spdl::cuda
