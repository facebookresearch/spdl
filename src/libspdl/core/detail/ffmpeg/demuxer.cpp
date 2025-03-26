/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/demuxer.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>

#define POS_INF std::numeric_limits<double>::infinity()
#define NEG_INF -std::numeric_limits<double>::infinity()

namespace spdl::core::detail {
namespace {

void init_fmt_ctx(AVFormatContext* fmt_ctx, DataInterface* di) {
  TRACE_EVENT("demuxing", "avformat_find_stream_info");
  CHECK_AVERROR(
      avformat_find_stream_info(di->get_fmt_ctx(), nullptr),
      fmt::format("Failed to find stream information: {}.", di->get_src()));
}

void enable_for_stream(AVFormatContext* fmt_ctx, int idx) {
  // Disable other streams
  fmt_ctx->streams[idx]->discard = AVDISCARD_DEFAULT;
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

AVStream*
get_stream(MediaType media_type, AVFormatContext* fmt_ctx, DataInterface* di) {
  AVMediaType type = [&]() {
    switch (media_type) {
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
  return fmt_ctx->streams[idx];
}
} // namespace

DemuxerImpl::DemuxerImpl(std::unique_ptr<DataInterface> di_)
    : di(std::move(di_)) {
  fmt_ctx = di->get_fmt_ctx();
  init_fmt_ctx(fmt_ctx, di.get());
}

DemuxerImpl::~DemuxerImpl() {
  TRACE_EVENT("demuxing", "Demuxer::~Demuxer");
  di.reset();
  // Technically, this is not necessary, but doing it here puts
  // the destruction of AVFormatContext under ~StreamingDemuxe, which
  // makes the trace easier to interpret.
}

template <MediaType media_type>
Codec<media_type> DemuxerImpl::get_default_codec() const {
  auto* stream = get_stream(media_type, fmt_ctx, di.get());
  auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
  return Codec<media_type>{
      stream->codecpar,
      {stream->time_base.num, stream->time_base.den},
      {frame_rate.num, frame_rate.den}};
}

template Codec<MediaType::Audio>
DemuxerImpl::get_default_codec<MediaType::Audio>() const;
template Codec<MediaType::Video>
DemuxerImpl::get_default_codec<MediaType::Video>() const;
template Codec<MediaType::Image>
DemuxerImpl::get_default_codec<MediaType::Image>() const;

bool DemuxerImpl::has_audio() const {
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (AVMEDIA_TYPE_AUDIO == fmt_ctx->streams[i]->codecpar->codec_type) {
      return true;
    }
  }
  return false;
}

Generator<AVPacketPtr> DemuxerImpl::demux() {
  int errnum = 0;
  while (errnum >= 0) {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    {
      TRACE_EVENT("demuxing", "av_read_frame");
      errnum = av_read_frame(fmt_ctx, packet.get());
    }
    if (errnum < 0 && errnum != AVERROR_EOF) {
      CHECK_AVERROR_NUM(
          errnum, fmt::format("Failed to read a packet. ({})", fmt_ctx->url));
    }
    if (errnum == AVERROR_EOF) {
      break;
    }
    co_yield std::move(packet);
  }
}

Generator<AVPacketPtr> DemuxerImpl::demux_window(
    AVStream* stream,
    const double end,
    std::optional<BitStreamFilter>& filter) {
  auto demuxing = this->demux();
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

template <MediaType media_type>
PacketsPtr<media_type> DemuxerImpl::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf) {
  TRACE_EVENT("demuxing", "detail::demux_window");

  auto [start, end] =
      window ? *window : std::tuple<double, double>{NEG_INF, POS_INF};
  if constexpr (media_type == MediaType::Video) {
    // Note:
    // Since the video frames can be non-chronological order, so we add small
    // margin to end
    end += 0.3;
  }

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

  auto stream = get_stream(media_type, fmt_ctx, di.get());
  enable_for_stream(fmt_ctx, stream->index);

  auto filter = [&]() -> std::optional<BitStreamFilter> {
    if (!bsf) {
      return std::nullopt;
    }
    return BitStreamFilter{*bsf, stream->codecpar};
  }();

  auto ret = std::make_unique<DemuxedPackets<media_type>>(
      di->get_src(),
      Codec<media_type>{
          bsf ? filter->get_output_codec_par() : stream->codecpar,
          Rational{stream->time_base.num, stream->time_base.den}},
      window);

  auto demuxing = this->demux_window(stream, end, filter);
  while (demuxing) {
    ret->push(demuxing().release());
    if constexpr (media_type == MediaType::Image) {
      break;
    }
  }
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    ret->codec.frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return ret;
}

template PacketsPtr<MediaType::Audio> DemuxerImpl::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> DemuxerImpl::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> DemuxerImpl::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template <MediaType media_type>
Generator<PacketsPtr<media_type>> DemuxerImpl::streaming_demux(
    int num_packets,
    const std::optional<std::string> bsf) {
  auto stream = get_stream(media_type, fmt_ctx, di.get());
  enable_for_stream(fmt_ctx, stream->index);

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
  auto make_packets =
      [&](std::vector<AVPacketPtr>&& pkts) -> PacketsPtr<media_type> {
    auto ret = std::make_unique<DemuxedPackets<media_type>>(
        di->get_src(),
        Codec<media_type>{
            bsf ? filter->get_output_codec_par() : stream->codecpar,
            Rational{stream->time_base.num, stream->time_base.den},
            frame_rate});
    for (auto& p : pkts) {
      ret->push(p.release());
    }
    return ret;
  };

  auto demuxing = this->demux_window(stream, POS_INF, filter);
  std::vector<AVPacketPtr> packets;
  packets.reserve(num_packets);
  while (demuxing) {
    packets.push_back(demuxing());
    if (packets.size() >= num_packets) {
      co_yield make_packets(std::move(packets));

      packets = std::vector<AVPacketPtr>();
      packets.reserve(num_packets);
    }
  }

  if (packets.size() > 0) {
    co_yield make_packets(std::move(packets));
  }
}

template Generator<PacketsPtr<MediaType::Audio>> DemuxerImpl::streaming_demux(
    int num_packets,
    const std::optional<std::string> bsf);
template Generator<PacketsPtr<MediaType::Video>> DemuxerImpl::streaming_demux(
    int num_packets,
    const std::optional<std::string> bsf);

} // namespace spdl::core::detail
