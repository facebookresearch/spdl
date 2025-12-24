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

#include <libspdl/core/bsf.h>
#include <libspdl/core/demuxing.h>

#include <fmt/core.h>
#include <set>

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

NvDecDecoder::NvDecDecoder() : core_(new detail::NvDecDecoderCore()) {}

NvDecDecoder::~NvDecDecoder() {
  if (core_) {
    delete core_;
  }
}

void NvDecDecoder::reset() {
  core_->reset();
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
  core_->init(cuda_config, codec, crop, width, height);
}

std::vector<CUDABuffer> NvDecDecoder::decode(
    spdl::core::VideoPacketsPtr packets) {
  std::vector<CUDABuffer> ret;
  core_->decode_packets(packets.get(), &ret);
  return ret;
}

std::vector<CUDABuffer> NvDecDecoder::flush() {
  std::vector<CUDABuffer> ret;
  core_->flush(&ret);
  return ret;
}

CUDABuffer NvDecDecoder::decode_all(spdl::core::VideoPacketsPtr packets) {
  return core_->decode_all(packets.get());
}

#ifdef SPDL_USE_NVCODEC
FrameBatchGenerator streaming_load_video_nvdec(
    spdl::core::Demuxer* demuxer,
    NvDecDecoder* decoder,
    spdl::core::BSF<spdl::core::MediaType::Video>* bsf,
    size_t num_frames) {
  TRACE_EVENT("nvdec", "streaming_load_video_nvdec");

  // Get video stream index
  int video_stream_index =
      demuxer->get_default_stream_index<spdl::core::MediaType::Video>();

  // Get packet stream from demuxer
  std::set<int> indices{video_stream_index};
  auto packet_gen = demuxer->streaming_demux(indices, num_frames, 0.0);

  // Buffer for accumulating decoded frames
  std::vector<CUDABuffer> buffers;

  // Process packets from demuxer
  for (auto& packets_map : packet_gen) {
    // Extract video packets for the video stream
    auto it = packets_map.find(video_stream_index);
    if (it == packets_map.end()) {
      continue;
    }

    auto packets = std::move(std::get<spdl::core::VideoPacketsPtr>(it->second));

    // Apply bitstream filter if provided
    if (bsf) {
      auto filtered = bsf->filter(std::move(packets), false);
      if (!filtered.has_value()) {
        continue;
      }
      packets = std::move(*filtered);
    }

    // Decode packets
    auto decoded = decoder->decode(std::move(packets));
    buffers.insert(buffers.end(), decoded.begin(), decoded.end());

    // Yield when we have enough frames
    while (buffers.size() >= num_frames) {
      std::vector<CUDABuffer> batch(
          buffers.begin(), buffers.begin() + num_frames);
      buffers.erase(buffers.begin(), buffers.begin() + num_frames);
      co_yield std::move(batch);
    }
  }

  // Flush BSF if provided
  if (bsf) {
    auto flushed = bsf->flush();
    if (flushed.has_value()) {
      const auto& pkt_series = (*flushed)->pkts;
      if (!pkt_series.get_packets().empty()) {
        auto decoded = decoder->decode(std::move(*flushed));
        buffers.insert(buffers.end(), decoded.begin(), decoded.end());
      }
    }
  }

  // Flush decoder
  auto flushed_frames = decoder->flush();
  buffers.insert(buffers.end(), flushed_frames.begin(), flushed_frames.end());

  // Yield remaining buffers in chunks
  while (buffers.size() >= num_frames) {
    std::vector<CUDABuffer> batch(
        buffers.begin(), buffers.begin() + num_frames);
    buffers.erase(buffers.begin(), buffers.begin() + num_frames);
    co_yield std::move(batch);
  }

  // Yield any remaining buffers
  if (!buffers.empty()) {
    co_yield std::move(buffers);
  }
}
#else
FrameBatchGenerator streaming_load_video_nvdec(
    spdl::core::Demuxer*,
    NvDecDecoder*,
    spdl::core::BSF<spdl::core::MediaType::Video>*,
    size_t) {
  NOT_SUPPORTED_NVCODEC;
}
#endif

} // namespace spdl::cuda
