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

////////////////////////////////////////////////////////////////////////////////
// StreamingDemuxer
////////////////////////////////////////////////////////////////////////////////
class StreamingDemuxer {
  Generator<std::map<int, AnyPackets>> gen_;

 public:
  StreamingDemuxer(
      detail::DemuxerImpl* p,
      const std::set<int>& stream_index,
      int num_packets,
      double duration);
  bool done();
  std::map<int, AnyPackets> next();
};

using StreamingDemuxerPtr = std::unique_ptr<StreamingDemuxer>;

////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////

class Demuxer {
  detail::DemuxerImpl* pImpl_;

 public:
  explicit Demuxer(DataInterfacePtr di);

  ~Demuxer();

  bool has_audio() const;

  // Get the codec of the default stream of the given media type
  template <MediaType media>
  Codec<media> get_default_codec() const;

  template <MediaType media>
  int get_default_stream_index() const;

  template <MediaType media>
  PacketsPtr<media> demux_window(
      const std::optional<TimeWindow>& window = std::nullopt,
      const std::optional<std::string>& bsf = std::nullopt);

  StreamingDemuxerPtr streaming_demux(
      const std::set<int>& indices,
      int num_packets,
      double duration);
};

using DemuxerPtr = std::unique_ptr<Demuxer>;

// Create a demuxer from an URI (file path, http, etc.)
DemuxerPtr make_demuxer(
    const std::string& src,
    const SourceAdaptorPtr& adaptor = nullptr,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

// Create a demuxer from an externally managed in-memory data
DemuxerPtr make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

} // namespace spdl::core
