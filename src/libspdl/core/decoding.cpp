/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoding.h>
#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/decoder.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

namespace spdl::core {
template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc) {
  TRACE_EVENT(
      "decoding",
      "decode_packets_ffmpeg",
      perfetto::Flow::ProcessScoped(packets->id));
  detail::Decoder decoder{packets->codecpar, packets->time_base, cfg};
  auto filter = detail::get_filter<media_type>(
      decoder.codec_ctx.get(), filter_desc, packets->frame_rate);
  auto ret = std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);

  auto gen = detail::decode_packets(packets->get_packets(), decoder, filter);
  while (gen) {
    ret->push_back(gen().release());
  }
  if (filter) {
    ret->time_base = filter->get_sink_time_base();
  }
  return ret;
}

template FFmpegAudioFramesPtr decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

template FFmpegVideoFramesPtr decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

template FFmpegImageFramesPtr decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc);

} // namespace spdl::core
