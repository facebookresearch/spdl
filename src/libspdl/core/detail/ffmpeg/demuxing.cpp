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

AVStream* init_fmt_ctx(AVFormatContext* fmt_ctx, enum MediaType type_) {
  {
    TRACE_EVENT("demuxing", "avformat_find_stream_info");
    CHECK_AVERROR(
        avformat_find_stream_info(fmt_ctx, nullptr),
        fmt::format("Failed to find stream information: {}.", fmt_ctx->url));
  }

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

std::tuple<double, double> NO_WINDOW{
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};

template <MediaType media_type>
PacketsPtr<media_type> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window = std::nullopt) {
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

  auto ret = std::make_unique<DemuxedPackets<media_type>>(
      fmt_ctx->url,
      stream->codecpar,
      Rational{stream->time_base.num, stream->time_base.den});
  ret->timestamp = window;

  auto demuxer = detail::Demuxer{fmt_ctx};
  auto demuxing = demuxer.demux();
  while (demuxing) {
    auto packet = demuxing();
    if (packet->stream_index != stream->index) {
      continue;
    }
    if (packet->pts * av_q2d(stream->time_base) > end) {
      break;
    }
    ret->push(packet.release());
  }
  return ret;
}

template PacketsPtr<MediaType::Audio> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window);

template PacketsPtr<MediaType::Video> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window);

template PacketsPtr<MediaType::Image> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window);

std::unique_ptr<DataInterface> get_interface(
    const std::string_view src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  if (!adaptor) {
    thread_local auto p = std::make_shared<SourceAdaptor>();
    return p->get(src, dmx_cfg.value_or(DemuxConfig{}));
  }
  return adaptor->get(src, dmx_cfg.value_or(DemuxConfig{}));
}

std::unique_ptr<DataInterface> get_in_memory_interface(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  thread_local SourceAdaptorPtr adaptor{new BytesAdaptor()};
  return get_interface(data, adaptor, dmx_cfg);
}

} // namespace spdl::core::detail
