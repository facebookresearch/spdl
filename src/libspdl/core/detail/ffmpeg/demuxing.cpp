/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/demuxer.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <cmath>
#include <optional>

namespace spdl::core::detail {

void init_fmt_ctx(DataInterface* di) {
  TRACE_EVENT("demuxing", "avformat_find_stream_info");
  CHECK_AVERROR(
      avformat_find_stream_info(di->get_fmt_ctx(), nullptr),
      fmt::format("Failed to find stream information: {}.", di->get_src()));
}

AVStream* get_stream(DataInterface* di, enum MediaType type_) {
  AVMediaType type = [&]() {
    switch (type_) {
      case MediaType::Audio:
        return AVMEDIA_TYPE_AUDIO;
      case MediaType::Image:
        [[fallthrough]];
      case MediaType::Video:
        return AVMEDIA_TYPE_VIDEO;
      default:
        SPDL_FAIL("Unexpected media type.");
    }
  }();
  auto* fmt_ctx = di->get_fmt_ctx();
  int idx;
  {
    TRACE_EVENT("demuxing", "av_find_best_stream");
    idx = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
  }
  if (idx < 0) {
    SPDL_FAIL(fmt::format(
        "No {} stream was found in {}.",
        av_get_media_type_string(type),
        di->get_src()));
  }
  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
  return fmt_ctx->streams[idx];
}

Generator<AVPacketPtr> demux_and_filter(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    std::optional<BitStreamFilter>& filter,
    double end = std::numeric_limits<double>::infinity()) {
  auto demuxer = Demuxer{fmt_ctx};
  auto demuxing = demuxer.demux();
  while (demuxing) {
    auto packet = demuxing();
    if (packet->stream_index != stream->index) {
      continue;
    }
    if (!filter) {
      if (packet->pts * av_q2d(stream->time_base) > end) {
        co_return;
      }
      co_yield std::move(packet);
    } else {
      auto filtering = filter->filter(packet.get());
      while (filtering) {
        auto filtered = filtering();
        if (filtered->pts * av_q2d(stream->time_base) > end) {
          co_return;
        }
        co_yield std::move(filtered);
      }
    }
  }
}

std::tuple<double, double> NO_WINDOW{
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};

template <MediaType media_type>
PacketsPtr<media_type> demux_window(
    DataInterface* di,
    const std::optional<std::tuple<double, double>>& window = std::nullopt,
    const std::optional<std::string>& bsf = std::nullopt) {
  TRACE_EVENT("demuxing", "detail::demux_window");
  auto [start, end] = window ? *window : NO_WINDOW;

  auto stream = get_stream(di, media_type);

  if constexpr (media_type == MediaType::Video) {
    // Note:
    // Since the video frames can be non-chronological order, so we add small
    // margin to end
    end += 0.3;
  }

  auto* fmt_ctx = di->get_fmt_ctx();
  if (!std::isinf(start)) {
    int64_t t = static_cast<int64_t>(start * AV_TIME_BASE);
    {
      TRACE_EVENT("demuxing", "av_seek_frame");
      CHECK_AVERROR(
          av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
          "Failed to seek to the timestamp {} [sec]",
          start);
    }
  }

  auto filter = [&]() -> std::optional<BitStreamFilter> {
    if (!bsf) {
      return std::nullopt;
    }
    return BitStreamFilter{*bsf, stream->codecpar};
  }();

  auto ret = std::make_unique<DemuxedPackets<media_type>>(
      di->get_src(),
      bsf ? filter->get_output_codec_par() : stream->codecpar,
      Rational{stream->time_base.num, stream->time_base.den});
  ret->timestamp = window;

  auto demuxing = demux_and_filter(fmt_ctx, stream, filter, end);
  while (demuxing) {
    ret->push(demuxing().release());
    if constexpr (media_type == MediaType::Image) {
      break;
    }
  }
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    ret->frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return ret;
}

template PacketsPtr<MediaType::Audio> demux_window(
    DataInterface* fmt_ctx,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> demux_window(
    DataInterface* fmt_ctx,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> demux_window(
    DataInterface* fmt_ctx,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template <MediaType media_type>
Generator<PacketsPtr<media_type>> streaming_demux(
    DataInterface* di,
    int num_packets,
    // Note: This is generator, so pass by value, not by reference.
    const std::optional<std::string> bsf) {
  auto* fmt_ctx = di->get_fmt_ctx();
  auto stream = get_stream(di, media_type);

  auto filter = [&]() -> std::optional<BitStreamFilter> {
    if (!bsf) {
      return std::nullopt;
    }
    return BitStreamFilter{*bsf, stream->codecpar};
  }();

  Rational frame_rate;
  if constexpr (media_type == MediaType::Video) {
    auto fr = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    frame_rate = Rational{fr.num, fr.den};
  }
  auto make_packets = [&](std::vector<AVPacket*>&& pkts) {
    auto ret = std::make_unique<DemuxedPackets<media_type>>(
        di->get_src(),
        bsf ? filter->get_output_codec_par() : stream->codecpar,
        Rational{stream->time_base.num, stream->time_base.den},
        std::move(pkts));
    ret->frame_rate = frame_rate;
    return std::move(ret);
  };
  auto demuxing = demux_and_filter(fmt_ctx, stream, filter);
  std::vector<AVPacket*> packets;
  packets.reserve(num_packets);
  while (demuxing) {
    packets.push_back(demuxing().release());
    if (packets.size() >= num_packets) {
      co_yield make_packets(std::move(packets));

      packets = std::vector<AVPacket*>();
      packets.reserve(num_packets);
    }
  }

  if (packets.size() > 0) {
    co_yield make_packets(std::move(packets));
  }
}

template Generator<PacketsPtr<MediaType::Video>> streaming_demux(
    DataInterface* di,
    int num_packets,
    const std::optional<std::string> bsf);

} // namespace spdl::core::detail
