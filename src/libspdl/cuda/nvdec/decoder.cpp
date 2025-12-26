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

// Macro for hiding argument names if not using this module.
// So as to workaround -Werror=unused-variable
#ifdef SPDL_USE_NVCODEC
#define _(var_name) var_name
#else
#define _(var_name)
#endif

namespace spdl::cuda {

namespace {
void validate_nvdec_params(
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height) {
  if (cuda_device_index < 0) {
    SPDL_FAIL(
        fmt::format(
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
    SPDL_FAIL(
        fmt::format(
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

NvDecDecoder::NvDecDecoder()
#ifdef SPDL_USE_NVCODEC
    : core_(new detail::NvDecDecoderCore())
#endif
{
}

NvDecDecoder::~NvDecDecoder() {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  if (core_) {
    delete core_;
  }
#endif
}

void NvDecDecoder::reset() {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  core_->reset();
#endif
}

void NvDecDecoder::init_decoder(
    // device config
    const CUDAConfig& cuda_config,
    // Source codec information
    const spdl::core::VideoCodec& codec,
    // Post-decoding processing params
    CropArea crop,
    int width,
    int height) {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  validate_nvdec_params(cuda_config.device_index, crop, width, height);
  core_->init_decoder(cuda_config, codec, crop, width, height);
#endif
}

CUDABufferGenerator NvDecDecoder::flush() {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  TRACE_EVENT("nvdec", "flush");

  // Flush decoder
  core_->flush();

  // Yield all remaining batches from the frame buffer
  while (core_->has_batch_ready()) {
    co_yield core_->pop_batch();
  }
#endif
}

void NvDecDecoder::init_buffer(size_t num_frames) {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  core_->init_buffer(num_frames);
#endif
}

CUDABuffer NvDecDecoder::decode_packets(spdl::core::VideoPacketsPtr packets) {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  return core_->decode_packets(packets.get());
#endif
}

CUDABufferGenerator NvDecDecoder::streaming_decode_packets(
    spdl::core::VideoPacketsPtr _(packets)) {
#ifndef SPDL_USE_NVCODEC
  NOT_SUPPORTED_NVCODEC;
#else
  TRACE_EVENT("nvdec", "streaming_decode_packets");

  // Process packets one by one
  for (auto pkt : packets->pkts.iter_data()) {
    core_->decode_packet(pkt);

    // Check if buffer has batch ready and yield
    while (core_->has_batch_ready()) {
      co_yield core_->pop_batch();
    }
  }
#endif
}

} // namespace spdl::cuda
