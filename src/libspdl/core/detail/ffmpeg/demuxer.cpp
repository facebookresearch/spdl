/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/demuxer.h"

#include <libspdl/core/rational_utils.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>

#include <set>

#define POS_INF std::numeric_limits<double>::infinity()
#define NEG_INF -std::numeric_limits<double>::infinity()

namespace spdl::core::detail {
namespace {

void init_fmt_ctx(AVFormatContext* fmt_ctx) {
  TRACE_EVENT("demuxing", "avformat_find_stream_info");
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      fmt::format("Failed to find stream information: {}.", fmt_ctx->url))

  // Disable all the non-media streams
  for (int i = 0; i < (int)fmt_ctx->nb_streams; ++i) {
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
    if (i < 0 || (int)fmt_ctx->nb_streams <= i) {
      SPDL_FAIL(
          fmt::format(
              "Stream index must be in range of [0, {}). Found: {}",
              fmt_ctx->nb_streams,
              i));
    }
    auto t = fmt_ctx->streams[i]->codecpar->codec_type;
    if (!(t == AVMEDIA_TYPE_AUDIO || t == AVMEDIA_TYPE_VIDEO)) {
      SPDL_FAIL(
          fmt::format(
              "Only audio/video streams are supported. Stream index {} is {}.",
              i,
              av_get_media_type_string(t)));
    }
  }
  // Disable other streams
  for (int i = 0; i < (int)fmt_ctx->nb_streams; ++i) {
    if (indices.contains(i)) {
      fmt_ctx->streams[i]->discard = AVDISCARD_DEFAULT;
    } else {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

} // namespace

DemuxerImpl::DemuxerImpl(DataInterfacePtr di) : di_(std::move(di)) {
  fmt_ctx_ = di_->get_fmt_ctx();
  init_fmt_ctx(fmt_ctx_);
}

DemuxerImpl::~DemuxerImpl() {
  TRACE_EVENT("demuxing", "Demuxer::~Demuxer");
  di_.reset();
  // Technically, this is not necessary, but doing it here puts
  // the destruction of AVFormatContext under ~StreamingDemuxe, which
  // makes the trace easier to interpret.
}

template <MediaType media>
Codec<media> DemuxerImpl::get_default_codec() const {
  int i = get_default_stream_index(media);
  AVStream* stream = fmt_ctx_->streams[i];
  auto frame_rate = av_guess_frame_rate(fmt_ctx_, stream, nullptr);
  return Codec<media>{stream->codecpar, stream->time_base, frame_rate};
}

template Codec<MediaType::Audio>
DemuxerImpl::get_default_codec<MediaType::Audio>() const;
template Codec<MediaType::Video>
DemuxerImpl::get_default_codec<MediaType::Video>() const;
template Codec<MediaType::Image>
DemuxerImpl::get_default_codec<MediaType::Image>() const;

template <MediaType media>
int DemuxerImpl::get_default_stream_index() const {
  enum AVMediaType t = []() {
    if constexpr (media == MediaType::Audio) {
      return AVMEDIA_TYPE_AUDIO;
    }
    if constexpr (media == MediaType::Video || media == MediaType::Image) {
      return AVMEDIA_TYPE_VIDEO;
    }
    SPDL_FAIL_INTERNAL("Unexpected media type.");
  }();
  int i = av_find_best_stream(fmt_ctx_, t, -1, -1, nullptr, 0);
  CHECK_AVERROR_NUM(
      i, fmt::format("Failed to find an audio stream in {}", fmt_ctx_->url))
  return i;
}

template int DemuxerImpl::get_default_stream_index<MediaType::Audio>() const;
template int DemuxerImpl::get_default_stream_index<MediaType::Video>() const;
template int DemuxerImpl::get_default_stream_index<MediaType::Image>() const;

bool DemuxerImpl::has_audio() const {
  for (int i = 0; i < (int)fmt_ctx_->nb_streams; ++i) {
    if (AVMEDIA_TYPE_AUDIO == fmt_ctx_->streams[i]->codecpar->codec_type) {
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
    idx = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
  }
  if (idx < 0) {
    SPDL_FAIL(
        fmt::format(
            "No {} stream was found in {}.",
            av_get_media_type_string(type),
            fmt_ctx_->url));
  }
  return idx;
}
Generator<AVPacketPtr> DemuxerImpl::demux(
    int stream_index,
    std::optional<AVRational> seek_time) {
  if (seek_time && stream_index >= 0) {
    auto tb = fmt_ctx_->streams[stream_index]->time_base;
    auto t = av_rescale(seek_time->num, tb.den, seek_time->den * tb.num);
    auto start = to_double(*seek_time);
    {
      TRACE_EVENT("demuxing", "av_seek_frame");
      CHECK_AVERROR(
          av_seek_frame(fmt_ctx_, stream_index, t - 1, AVSEEK_FLAG_BACKWARD),
          "Failed to seek to the timestamp {} ({}/{}) [sec]",
          start,
          seek_time->num,
          seek_time->den);
    }
  }

  int errnum = 0;
  bool is_first_packet = (seek_time && stream_index >= 0);

  while (errnum >= 0) {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    {
      TRACE_EVENT("demuxing", "av_read_frame");
      errnum = av_read_frame(fmt_ctx_, packet.get());
    }
    if (errnum < 0 && errnum != AVERROR_EOF) {
      CHECK_AVERROR_NUM(
          errnum, fmt::format("Failed to read a packet. ({})", fmt_ctx_->url))
    }
    if (errnum == AVERROR_EOF) {
      break;
    }

    // Handle RASL frames on best-effort basis
    //
    // RASL (Random Access Skipped Leading) frames in HEVC (H.265) refers to
    // special pictures that appear after a key random access point
    // (like a CRA picture) but depend on pictures that come before the random
    // access point in coding order, making them undecodable if you start from
    // the CRA picture; they are typically skipped during playback to enable
    // clean, immediate starts at CRA points, unlike RADL (Random Access
    // Decodable Leading) pictures which are decodable.
    //
    // Here, if the first packet has PTS higher than seek time, it might be
    // RASL. So we seek again using DTS - 1
    if (is_first_packet) {
      is_first_packet = false;
      auto tb = fmt_ctx_->streams[stream_index]->time_base;
      auto t = av_rescale(seek_time->num, tb.den, seek_time->den * tb.num);

      // If packet PTS is higher than seek time, it might be RASL.
      // seek again using DTS - 1
      if (packet->pts > t) {
        auto dts_seek_target = packet->dts - 1;
        {
          TRACE_EVENT("demuxing", "av_seek_frame (retry)");
          CHECK_AVERROR(
              av_seek_frame(
                  fmt_ctx_,
                  stream_index,
                  dts_seek_target,
                  AVSEEK_FLAG_BACKWARD),
              "Failed to seek to DTS-based timestamp {} [{}/{}}]",
              dts_seek_target,
              tb.den,
              tb.num);
        }
        // Continue to read from the new position
        continue;
      }
    }

    co_yield std::move(packet);
  }
}

Generator<AVPacketPtr> DemuxerImpl::demux_window(
    AVStream* stream,
    const AVRational end,
    std::optional<BSFImpl>& filter,
    std::optional<AVRational> seek_time) {
  for (auto&& packet : this->demux(stream->index, seek_time)) {
    if (packet->stream_index != stream->index) {
      continue;
    }
    if (!filter) {
      auto pts = to_rational(packet->pts, stream->time_base);
      if (av_cmp_q(pts, end) >= 0) {
        co_return;
      }
      co_yield std::move(packet);
    } else {
      for (auto&& filtered : filter->filter(packet.get())) {
        auto pts = to_rational(filtered->pts, stream->time_base);
        if (av_cmp_q(pts, end) >= 0) {
          co_return;
        }
        co_yield std::move(filtered);
      }
    }
  }
}

template <MediaType media>
PacketsPtr<media> DemuxerImpl::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf) {
  TRACE_EVENT("demuxing", "detail::demux_window");

  auto index = get_default_stream_index(media);
  enable_for_stream(fmt_ctx_, {index});
  auto* stream = fmt_ctx_->streams[index];

  std::optional<AVRational> seek_time = std::nullopt;
  if (window) {
    seek_time = std::get<0>(*window);
  }

  auto filter = bsf ? std::optional<BSFImpl>{BSFImpl{*bsf, stream->codecpar}}
                    : std::nullopt;

  auto ret = std::make_unique<Packets<media>>(
      fmt_ctx_->url,
      stream->index,
      Codec<media>{
          bsf ? filter->get_output_codec_par() : stream->codecpar,
          bsf ? filter->get_output_time_base() : stream->time_base,
          av_guess_frame_rate(fmt_ctx_, stream, nullptr)},
      window);

  AVRational end = {1, 0};
  if (window) {
    end = std::get<1>(*window);
    if constexpr (media == MediaType::Video) {
      // Note:
      // Since the video frames can be non-chronological order,
      // we add small margin to the end
      end = av_add_q(end, {3, 10});
    }
  }
  for (auto&& packet : this->demux_window(stream, end, filter, seek_time)) {
    ret->pkts.push(packet.release());
  }
  return ret;
}

template PacketsPtr<MediaType::Audio> DemuxerImpl::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> DemuxerImpl::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> DemuxerImpl::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

namespace {
AnyPackets
mk_packets(AVStream* stream, const char* src, std::vector<AVPacketPtr>&& pkts) {
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
  SPDL_FAIL(
      fmt::format(
          "Unexpected media type was provided: {}",
          av_get_media_type_string(stream->codecpar->codec_type)));
}
} // namespace

Generator<std::map<int, AnyPackets>> DemuxerImpl::streaming_demux(
    const std::set<int> stream_indices,
    int num_packets,
    double duration) {
  if (num_packets <= 0 && duration <= 0) {
    SPDL_FAIL("Either `duration` or `num_packets` must be specified.");
  }
  if (num_packets > 0 && duration > 0) {
    SPDL_FAIL("Only one of `duration` or `num_packets` can be specified.");
  }

  enable_for_stream(fmt_ctx_, stream_indices);
  std::map<int, AnyPackets> ret;

  // Some samples can start from negative PTS, like -0.023, so we use much
  // bigger number.
  double t0 = -100;

#define YIELD                        \
  co_yield std::move(ret);           \
  ret = std::map<int, AnyPackets>{}; \
  t0 = pts

  for (auto&& packet : this->demux()) {
    auto i = packet->stream_index;
    auto* stream = fmt_ctx_->streams[i];
    double pts = to_double(to_rational(packet->pts, stream->time_base));
    if (t0 < -99) {
      t0 = pts;
    }

    if (duration > 0 && (pts - t0 > duration)) {
      YIELD;
    }

    if (stream_indices.contains(i)) {
      if (!ret.contains(i)) {
        ret.emplace(i, mk_packets(stream, fmt_ctx_->url, {}));
      }
      std::visit([&](auto& v) { v->pkts.push(packet.release()); }, ret[i]);
      int num_pkts = std::visit(
          [&](auto& v) -> int { return (int)v->pkts.get_packets().size(); },
          ret[i]);
      if (num_packets > 0 && num_pkts >= num_packets) {
        YIELD;
      }
    }
  }
  if (ret.size()) {
    co_yield std::move(ret);
  }
}

} // namespace spdl::core::detail
