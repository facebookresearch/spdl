/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/types.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

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

// Used to expose the row data without exposing AVPacket* struct
// This allows NVDEC decoder to not include the libavutil header.
struct RawPacketData {
  uint8_t* data;
  int size;
  int64_t pts;
};

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

  // Codec info necessary for decoding
  Codec<media_type> codec;

 private:
  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

 public:
  DemuxedPackets(
      std::string src,
      Codec<media_type>&& codec,
      std::optional<std::tuple<double, double>> timestamp = std::nullopt);

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

  // Get the PTS of the specified frame.
  // throws if the index is not within the range
  int64_t get_pts(size_t index = 0) const;

  int get_num_channels() const
    requires(media_type == MediaType::Audio);

  int get_sample_rate() const
    requires(media_type == MediaType::Audio);

  Codec<media_type> get_codec() const;

  Generator<RawPacketData> iter_packets() const;

  std::string get_summary() const;
};

template <MediaType media_type>
PacketsPtr<media_type> clone(const DemuxedPackets<media_type>& src);

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices);

} // namespace spdl::core
