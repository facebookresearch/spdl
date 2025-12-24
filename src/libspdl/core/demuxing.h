/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/codec.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <set>
#include <string_view>
#include <tuple>

namespace spdl::core {
namespace detail {
class DemuxerImpl;
} // namespace detail

/// Generator type for streaming demux operations.
using AnyPacketsGenerator = Generator<std::map<int, AnyPackets>>;

////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////

/// Media demuxer for extracting packets from containers.
///
/// Demuxer reads media files or streams and extracts compressed packets
/// for individual streams (audio, video, etc.). It supports both one-shot
/// demuxing (demux_window) and streaming demuxing for large files.
class Demuxer {
  detail::DemuxerImpl* pImpl_;

 public:
  /// Construct a demuxer from a data interface.
  ///
  /// @param di Data interface providing access to media data.
  explicit Demuxer(DataInterfacePtr di);

  /// Destructor.
  ~Demuxer();

  /// Check if the media contains audio streams.
  ///
  /// @return true if audio is present, false otherwise.
  bool has_audio() const;

  /// Get the codec of the default stream for the given media type.
  ///
  /// @tparam media Media type (Audio, Video, or Image).
  /// @return Codec information for the default stream.
  template <MediaType media>
  Codec<media> get_default_codec() const;

  /// Get the default stream index for the given media type.
  ///
  /// @tparam media Media type (Audio, Video, or Image).
  /// @return Stream index of the default stream.
  template <MediaType media>
  int get_default_stream_index() const;

  /// Demux a time window from the default stream.
  ///
  /// @tparam media Media type (Audio, Video, or Image).
  /// @param window Optional time window to extract. If not specified, extracts
  /// all packets.
  /// @param bsf Optional bitstream filter to apply.
  /// @return Extracted packets.
  template <MediaType media>
  PacketsPtr<media> demux_window(
      const std::optional<TimeWindow>& window = std::nullopt,
      const std::optional<std::string>& bsf = std::nullopt);

  /// Create a streaming demuxer for specified streams.
  ///
  /// @param indices Set of stream indices to demux.
  /// @param num_packets Number of packets to extract per iteration.
  /// @param duration Duration in seconds to extract per iteration.
  /// @return Generator that yields packets from the specified streams.
  AnyPacketsGenerator streaming_demux(
      const std::set<int>& indices,
      int num_packets,
      double duration);
};

/// Unique pointer to a Demuxer instance.
using DemuxerPtr = std::unique_ptr<Demuxer>;

/// Create a demuxer from a URI (file path, HTTP URL, etc.).
///
/// @param src Source URI string.
/// @param adaptor Optional source adaptor for custom data sources.
/// @param dmx_cfg Optional demuxer configuration.
/// @return Demuxer instance.
DemuxerPtr make_demuxer(
    const std::string& src,
    const SourceAdaptorPtr& adaptor = nullptr,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt,
    const std::optional<std::string>& name = std::nullopt);

/// Create a demuxer from externally managed in-memory data.
///
/// @param data String view of the media data.
/// @param dmx_cfg Optional demuxer configuration.
/// @return Demuxer instance.
DemuxerPtr make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt,
    const std::optional<std::string>& name = std::nullopt);

} // namespace spdl::core
