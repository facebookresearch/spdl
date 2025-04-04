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
#include <variant>
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

using AnyPackets =
    std::variant<AudioPacketsPtr, VideoPacketsPtr, ImagePacketsPtr>;

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
  explicit PacketSeries(const PacketSeries&);
  PacketSeries& operator=(const PacketSeries&);
  PacketSeries(PacketSeries&& other) noexcept;
  PacketSeries& operator=(PacketSeries&& other) noexcept;

  // Add a new AVPacket
  void push(AVPacket*);
  // Iterate through the packets
  const std::vector<AVPacket*>& get_packets() const;
  Generator<RawPacketData> iter_data() const;
};

// Struct passed from IO thread pool to decoder thread pool.
// Similar to Frames, AVFrame pointers are bulk released.
// Optionally, it carries codec information for decoding and bitstream filtering
template <MediaType media>
struct Packets {
  uintptr_t id{};
  std::string src;

  PacketSeries pkts;
  Rational time_base{};
  std::optional<std::tuple<double, double>> timestamp;

  // Code should be available only when it's demuxed by `demux_window`
  // method. When streaming demuxing, codec object must be fetched from
  // demuxer directly.
  std::optional<Codec<media>> codec;

  Packets() = default;

  // Constructing Packets from demuxer for one-time decoding
  // Need codec to initialize the decoder.
  // Optionally remember the timestamp user asked.
  Packets(
      const std::string& src,
      Codec<media>&& codec,
      const std::optional<std::tuple<double, double>>& timestamp = {});

  // Constructing Packets from demuxer for streaming
  // No need for codec and time base, as decoder is initialized
  // seprately.
  explicit Packets(const std::string& src);

  // Constructing Packets from encoder for muxing
  Packets(uintptr_t id, Rational time_base);

  explicit Packets(const Packets<media>&);
  Packets<media>& operator=(const Packets<media>&);
  Packets(Packets<media>&&) noexcept;
  Packets<media>& operator=(Packets<media>&&) noexcept;
  ~Packets() = default;
};

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices);

} // namespace spdl::core
