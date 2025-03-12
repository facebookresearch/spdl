/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string_view>
#include <tuple>

struct AVFormatContext;
struct AVStream;

namespace spdl::core {

template <MediaType media_type>
class StreamingDemuxer {
  Generator<PacketsPtr<media_type>> gen;

 public:
  StreamingDemuxer(
      DataInterface* di,
      int num_packets,
      const std::optional<std::string>& bsf);
  bool done();
  PacketsPtr<media_type> next();
};

template <MediaType media_type>
using StreamingDemuxerPtr = std::unique_ptr<StreamingDemuxer<media_type>>;

class Demuxer {
  std::unique_ptr<DataInterface> di;
  AVFormatContext* fmt_ctx;

 public:
  explicit Demuxer(std::unique_ptr<DataInterface> di);

  ~Demuxer();

  bool has_audio();

  template <MediaType media_type>
  PacketsPtr<media_type> demux_window(
      const std::optional<std::tuple<double, double>>& window = std::nullopt,
      const std::optional<std::string>& bsf = std::nullopt);

  template <MediaType media_type>
  StreamingDemuxerPtr<media_type> stream_demux(
      int num_packets,
      const std::optional<std::string>& bsf = std::nullopt);
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

// Apply bitstream filter for NVDEC video decoding
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core
