/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

struct AVCodecParameters;
struct AVPacket;

namespace spdl::core {

template <MediaType media_type>
class DemuxedPackets;

using AudioPackets = DemuxedPackets<MediaType::Audio>;
using VideoPackets = DemuxedPackets<MediaType::Video>;
using ImagePackets = DemuxedPackets<MediaType::Image>;

template <MediaType media_type>
using PacketsPtr = std::unique_ptr<DemuxedPackets<media_type>>;

using AudioPacketsPtr = PacketsPtr<MediaType::Audio>;
using VideoPacketsPtr = PacketsPtr<MediaType::Video>;
using ImagePacketsPtr = PacketsPtr<MediaType::Image>;

// Struct passed from IO thread pool to decoder thread pool.
// Similar to FFmpegFrames, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
template <MediaType media_type>
class DemuxedPackets {
 public:
  uint64_t id;
  // Source information
  std::string src;
  std::optional<std::tuple<double, double>> timestamp;

  //
  AVCodecParameters* codecpar = nullptr;
  Rational time_base = {0, 1};

  // frame rate for video
  Rational frame_rate = {0, 1};

 private:
  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

 public:
  DemuxedPackets(
      std::string src,
      std::tuple<double, double> timestamp,
      AVCodecParameters* codecpar,
      Rational time_base);

  DemuxedPackets(
      std::string src,
      AVCodecParameters* codecpar,
      Rational time_base);

  DemuxedPackets(
      std::string src,
      AVCodecParameters* codecpar,
      Rational time_base,
      std::vector<AVPacket*>&& packets);

  // Destructor releases AVPacket* resources
  ~DemuxedPackets();
  // No copy/move constructors
  DemuxedPackets(const DemuxedPackets&) = delete;
  DemuxedPackets& operator=(const DemuxedPackets&) = delete;
  DemuxedPackets(DemuxedPackets&& other) noexcept = delete;
  DemuxedPackets& operator=(DemuxedPackets&& other) noexcept = delete;

  void push(AVPacket*);
  const std::vector<AVPacket*>& get_packets() const;
  const char* get_media_format_name() const;

  int get_width() const;
  int get_height() const;
  Rational get_frame_rate() const;

  // Get the number of valid packets, that is, the number of frames returned
  // by decoding function when decoded.
  // This is different from `packets.size()` when timestamp is set.
  // This is the number of packets visible to users.
  size_t num_packets() const
    requires(media_type == MediaType::Video || media_type == MediaType::Image);
};

template <MediaType media_type>
PacketsPtr<media_type> clone(const DemuxedPackets<media_type>& src);

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices);

} // namespace spdl::core
