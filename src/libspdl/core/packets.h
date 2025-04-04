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

template <MediaType media>
struct Packets;

using AudioPackets = Packets<MediaType::Audio>;
using VideoPackets = Packets<MediaType::Video>;
using ImagePackets = Packets<MediaType::Image>;

template <MediaType media>
using PacketsPtr = std::unique_ptr<Packets<media>>;

using AudioPacketsPtr = PacketsPtr<MediaType::Audio>;
using VideoPacketsPtr = PacketsPtr<MediaType::Video>;
using ImagePacketsPtr = PacketsPtr<MediaType::Image>;

class PacketSeries;
using PacketSeriesPtr = std::unique_ptr<PacketSeries>;

// Used to expose the row data without exposing AVPacket* struct
// This allows NVDEC decoder to not include the libavutil header.
struct RawPacketData {
  uint8_t* data;
  int size;
  int64_t pts;
};

// Helper struct which manages a series of AVPacket* pointers.
class PacketSeries {
  friend struct Packets<MediaType::Audio>;
  friend struct Packets<MediaType::Video>;
  friend struct Packets<MediaType::Image>;

  std::vector<AVPacket*> container = {};

 public:
  PacketSeries();

  // Destructor releases AVPacket* resources
  ~PacketSeries();
  PacketSeries(const PacketSeries&) = delete;
  PacketSeries& operator=(const PacketSeries&) = delete;
  PacketSeries(PacketSeries&& other) noexcept;
  PacketSeries& operator=(PacketSeries&& other) noexcept;
};

// Struct passed from IO thread pool to decoder thread pool.
// Similar to Frames, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
template <MediaType media>
struct Packets {
 public:
  uintptr_t id;
  std::string src;

 private:
  PacketSeries pkts;

 public:
  Rational time_base;
  std::optional<std::tuple<double, double>> timestamp;
  Codec<media> codec;

 public:
  Packets(
      const std::string& src,
      Codec<media>&& codec,
      const std::optional<std::tuple<double, double>>& timestamp =
          std::nullopt);

  const std::vector<AVPacket*>& get_packets() const;
  void push(AVPacket*);

  Generator<RawPacketData> iter_data() const;

  // Get the PTS of the specified frame.
  // throws if the index is not within the range
  int64_t get_pts(size_t index = 0) const;

  // Get the number of valid packets, that is, the number of frames returned
  // by decoding function when decoded.
  // This is different from `packets.size()` when timestamp is set.
  // This is the number of packets visible to users.
  size_t num_packets() const
    requires(media == MediaType::Video || media == MediaType::Image);

  PacketsPtr<media> clone() const;
};

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices);

} // namespace spdl::core
