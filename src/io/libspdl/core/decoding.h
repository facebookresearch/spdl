/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string>

namespace spdl::core {

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder;

template <MediaType media_type>
using DecoderPtr = std::unique_ptr<StreamingDecoder<media_type>>;

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder {
  struct Impl;
  Impl* pImpl;

  explicit StreamingDecoder(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig>& cfg = std::nullopt,
      const std::optional<std::string>& filter_desc = std::nullopt);

  ~StreamingDecoder();

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

template <MediaType media_type>
DecoderPtr<media_type> make_decoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::optional<std::string>& filter_desc);

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg = std::nullopt,
    const std::optional<std::string>& filter_desc = std::nullopt);

template <MediaType media_type>
CUDABufferPtr decode_packets_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt);

CUDABufferPtr decode_packets_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const CUDAConfig& cuda_config,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict);

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

} // namespace spdl::core
