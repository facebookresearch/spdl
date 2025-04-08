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

#include <set>

#define POS_INF std::numeric_limits<double>::infinity()
#define NEG_INF -std::numeric_limits<double>::infinity()

namespace spdl::core::detail {
namespace {

void init_fmt_ctx(AVFormatContext* fmt_ctx, DataInterface* di) {
  TRACE_EVENT("demuxing", "avformat_find_stream_info");
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      fmt::format("Failed to find stream information: {}.", di->get_src()));

  // Disable all the non-media streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    switch (fmt_ctx->streams[i]->codecpar->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
      case AVMEDIA_TYPE_VIDEO:
        break;
      default:
        fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

void enable_for_stream(AVFormatContext* fmt_ctx, const std::set<int>& indices) {
  for (auto i : indices) {
    if (i < 0 || fmt_ctx->nb_streams <= i) {
      SPDL_FAIL(fmt::format(
          "Stream index must be in range of [0, {}). Found: {}",
          fmt_ctx->nb_streams,
          i));
    }
    auto t = fmt_ctx->streams[i]->codecpar->codec_type;
    if (!(t == AVMEDIA_TYPE_AUDIO || t == AVMEDIA_TYPE_VIDEO)) {
      SPDL_FAIL(fmt::format(
          "Only audio/video streams are supported. Stream index {} is {}.",
          i,
          av_get_media_type_string(t)));
    }
  }
  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (indices.contains(i)) {
      fmt_ctx->streams[i]->discard = AVDISCARD_DEFAULT;
    } else {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

} // namespace

DemuxerImpl::DemuxerImpl(DataInterfacePtr di_) : di(std::move(di_)) {
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

template <MediaType media>
Codec<media> DemuxerImpl::get_default_codec() const {
  int i = get_default_stream_index(media);
  AVStream* stream = fmt_ctx->streams[i];
  auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
  return Codec<media>{stream->codecpar, stream->time_base, frame_rate};
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

int DemuxerImpl::get_default_stream_index(MediaType media) const {
  AVMediaType type = [&]() {
    switch (media) {
      case MediaType::Audio:
        return AVMEDIA_TYPE_AUDIO;
      case MediaType::Image:
        [[fallthrough]];
      case MediaType::Video:
        return AVMEDIA_TYPE_VIDEO;
      default:;
    }
    SPDL_FAIL("Unexpected media type.");
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
  return idx;
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
    std::optional<BSFImpl>& filter) {
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

template <MediaType media>
PacketsPtr<media> DemuxerImpl::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf) {
  TRACE_EVENT("demuxing", "detail::demux_window");

  auto [start, end] =
      window ? *window : std::tuple<double, double>{NEG_INF, POS_INF};
  if constexpr (media == MediaType::Video) {
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

  auto index = get_default_stream_index(media);
  enable_for_stream(fmt_ctx, {index});
  auto* stream = fmt_ctx->streams[index];

  auto filter = [&]() -> std::optional<BSFImpl> {
    if (!bsf) {
      return std::nullopt;
    }
    return BSFImpl{*bsf, stream->codecpar};
  }();

  Rational frame_rate{1, 1};
  if constexpr (media == MediaType::Video) {
    frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
  }
  auto ret = std::make_unique<Packets<media>>(
      di->get_src(),
      stream->index,
      Codec<media>{
          bsf ? filter->get_output_codec_par() : stream->codecpar,
          stream->time_base,
          frame_rate},
      window);

  auto demuxing = this->demux_window(stream, end, filter);
  while (demuxing) {
    ret->pkts.push(demuxing().release());
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

AnyPackets mk_packets(
    AVStream* stream,
    const std::string& src,
    std::vector<AVPacketPtr>&& pkts) {
  switch (stream->codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO: {
      auto ret =
          std::make_unique<AudioPackets>(src, stream->index, stream->time_base);
      for (auto& p : pkts) {
        ret->pkts.push(p.release());
      }
      return ret;
    }
    case AVMEDIA_TYPE_VIDEO: {
      auto ret =
          std::make_unique<VideoPackets>(src, stream->index, stream->time_base);
      for (auto& p : pkts) {
        ret->pkts.push(p.release());
      }
      return ret;
    }
    default:;
  }
  SPDL_FAIL(fmt::format(
      "Unexpected media type was provided: {}",
      av_get_media_type_string(stream->codecpar->codec_type)));
}

Generator<AnyPackets> DemuxerImpl::streaming_demux(
    int stream_index,
    int num_packets,
    const std::optional<std::string> bsf) {
  enable_for_stream(fmt_ctx, {stream_index});
  auto* stream = fmt_ctx->streams[stream_index];

  auto filter = [&]() -> std::optional<BSFImpl> {
    if (!bsf) {
      return std::nullopt;
    }
    return BSFImpl{*bsf, stream->codecpar};
  }();

  auto mkpkts = [&](std::vector<AVPacketPtr>&& pkts) {
    return mk_packets(stream, di->get_src(), std::move(pkts));
  };

  auto demuxing = this->demux_window(stream, POS_INF, filter);
  std::vector<AVPacketPtr> packets;
  packets.reserve(num_packets);
  while (demuxing) {
    packets.push_back(demuxing());
    if (packets.size() >= num_packets) {
      co_yield mkpkts(std::move(packets));

      packets = std::vector<AVPacketPtr>();
      packets.reserve(num_packets);
    }
  }

  if (packets.size() > 0) {
    co_yield mkpkts(std::move(packets));
  }
}

} // namespace spdl::core::detail
