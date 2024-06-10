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

void init_fmt_ctx(AVFormatContext* fmt_ctx) {
  TRACE_EVENT("demuxing", "avformat_find_stream_info");
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      fmt::format("Failed to find stream information: {}.", fmt_ctx->url));
}

AVStream* get_stream(AVFormatContext* fmt_ctx, enum MediaType type_) {
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
  int idx;
  {
    TRACE_EVENT("demuxing", "av_find_best_stream");
    idx = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
  }
  if (idx < 0) {
    SPDL_FAIL(fmt::format(
        "No {} stream was found in {}.",
        av_get_media_type_string(type),
        fmt_ctx->url));
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
    double end) {
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
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window = std::nullopt,
    const std::optional<std::string>& bsf = std::nullopt) {
  TRACE_EVENT("demuxing", "detail::demux_window");
  auto [start, end] = window ? *window : NO_WINDOW;

  if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
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

  auto filter = [&]() -> std::optional<BitStreamFilter> {
    if (!bsf) {
      return std::nullopt;
    }
    return BitStreamFilter{*bsf, stream->codecpar};
  }();

  auto ret = std::make_unique<DemuxedPackets<media_type>>(
      fmt_ctx->url,
      bsf ? filter->get_output_codec_par() : stream->codecpar,
      Rational{stream->time_base.num, stream->time_base.den});
  ret->timestamp = window;

  auto demuxing = demux_and_filter(fmt_ctx, stream, filter, end);
  while (demuxing) {
    ret->push(demuxing().release());
  }
  return ret;
}

template PacketsPtr<MediaType::Audio> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

} // namespace spdl::core::detail
