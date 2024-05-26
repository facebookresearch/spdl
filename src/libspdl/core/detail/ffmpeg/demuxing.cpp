#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

#include <cmath>

namespace spdl::core {
namespace detail {
////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////
namespace {
inline AVStream* init_fmt_ctx(AVFormatContext* fmt_ctx, enum MediaType type_) {
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
    const std::optional<std::tuple<double, double>>& window) {
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

  double packet_ts = -1;
  do {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    int errnum;
    {
      TRACE_EVENT("demuxing", "av_read_frame");
      errnum = av_read_frame(fmt_ctx, packet.get());
    }
    if (errnum == AVERROR_EOF) {
      break;
    }
    CHECK_AVERROR_NUM(errnum, "Failed to process packet.");
    if (packet->stream_index != stream->index) {
      continue;
    }
    packet_ts = packet->pts * av_q2d(stream->time_base);
    if (packet_ts <= end) {
      ret->push(packet.release());
    }
  } while (packet_ts < end);
  return ret;
}

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

} // namespace
} // namespace detail

template <MediaType media_type>
StreamingDemuxer<media_type>::StreamingDemuxer(
    const std::string uri,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg)
    : di(detail::get_interface(uri, adaptor, dmx_cfg)),
      fmt_ctx(di->get_fmt_ctx()),
      stream(detail::init_fmt_ctx(fmt_ctx, media_type)){};

template <MediaType media_type>
StreamingDemuxer<media_type>::StreamingDemuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg)
    : di(detail::get_in_memory_interface(data, dmx_cfg)),
      fmt_ctx(di->get_fmt_ctx()),
      stream(detail::init_fmt_ctx(fmt_ctx, media_type)) {}

template <MediaType media_type>
PacketsPtr<media_type> StreamingDemuxer<media_type>::demux_window(
    const std::optional<std::tuple<double, double>>& window) {
  auto packets = detail::demux_window<media_type>(fmt_ctx, stream, window);
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    packets->frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return packets;
}

template class StreamingDemuxer<MediaType::Audio>;
template class StreamingDemuxer<MediaType::Video>;
template class StreamingDemuxer<MediaType::Image>;

////////////////////////////////////////////////////////////////////////////////
// Demuxing for Image
////////////////////////////////////////////////////////////////////////////////
namespace detail {
namespace {
ImagePacketsPtr demux_image(AVFormatContext* fmt_ctx) {
  TRACE_EVENT("demuxing", "detail::demux");
  AVStream* stream = init_fmt_ctx(fmt_ctx, MediaType::Video);

  auto package = std::make_unique<DemuxedPackets<MediaType::Image>>(
      fmt_ctx->url, stream->codecpar, Rational{1, 1});

  int ite = 0;
  do {
    ++ite;
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    int errnum;
    {
      TRACE_EVENT("demuxing", "av_read_frame");
      errnum = av_read_frame(fmt_ctx, packet.get());
    }
    if (errnum == AVERROR_EOF) {
      break;
    }
    CHECK_AVERROR_NUM(
        errnum, fmt::format("Failed to process packet. {}", fmt_ctx->url));
    if (packet->stream_index != stream->index) {
      continue;
    }
    package->push(packet.release());
    break;
  } while (ite < 1000);
  if (!package->num_packets()) {
    SPDL_FAIL(
        fmt::format("Failed to demux a sigle frame from {}", fmt_ctx->url));
  }
  return package;
}

} // namespace
} // namespace detail

ImagePacketsPtr demux_image(
    const std::string uri,
    const SourceAdaptorPtr adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  auto interface = detail::get_interface(uri, adaptor, dmx_cfg);
  return detail::demux_image(interface->get_fmt_ctx());
}

ImagePacketsPtr demux_image(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool _zero_clear) {
  thread_local SourceAdaptorPtr adaptor{new BytesAdaptor()};
  auto interface = detail::get_interface(data, adaptor, dmx_cfg);
  auto result = detail::demux_image(interface->get_fmt_ctx());
  if (_zero_clear) {
    std::memset((void*)data.data(), 0, data.size());
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Bit Stream Filtering for NVDEC
////////////////////////////////////////////////////////////////////////////////
namespace detail {
namespace {
AVBSFContextPtr init_bsf(AVCodecParameters* codecpar) {
  // Note
  // FFmpeg's implementation applies BSF to all H264/HEVC formats,
  //
  // https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1185-L1191
  //
  // while NVidia SDK samples exclude those with the following substrings in
  // long_name attribute
  //
  //  "QuickTime / MOV", "FLV (Flash Video)", "Matroska / WebM"
  const char* name;
  switch (codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      name = "h264_mp4toannexb";
      break;
    case AV_CODEC_ID_HEVC:
      name = "hevc_mp4toannexb";
      break;
    default:
      return {nullptr};
  }

  TRACE_EVENT("demuxing", "init_bsf");
  const AVBitStreamFilter* bsf = av_bsf_get_by_name(name);
  if (!bsf) {
    SPDL_FAIL(fmt::format("Bit stream filter ({}) is not available", name));
  }
  AVBSFContext* p = nullptr;
  CHECK_AVERROR(av_bsf_alloc(bsf, &p), "Failed to allocate AVBSFContext.");
  AVBSFContextPtr bsf_ctx{p};
  CHECK_AVERROR(
      avcodec_parameters_copy(p->par_in, codecpar),
      "Failed to copy codec parameter.");
  CHECK_AVERROR(av_bsf_init(p), "Failed to initialize AVBSFContext..");
  return bsf_ctx;
}
AVPacketPtr apply_bsf(AVBSFContext* bsf_ctx, AVPacket* packet) {
  AVPacketPtr filtered{CHECK_AVALLOCATE(av_packet_alloc())};
  {
    TRACE_EVENT("decoding", "av_bsf_send_packet");
    CHECK_AVERROR(
        av_bsf_send_packet(bsf_ctx, packet),
        "Failed to send packet to bit stream filter.");
  }
  {
    TRACE_EVENT("decoding", "av_bsf_receive_packet");
    CHECK_AVERROR(
        av_bsf_receive_packet(bsf_ctx, filtered.get()),
        "Failed to fetch packet from bit stream filter.");
  }
  return filtered;
}
} // namespace
} // namespace detail

VideoPacketsPtr apply_bsf(VideoPacketsPtr packets) {
  TRACE_EVENT("demuxing", "apply_bsf");
  auto bsf_ctx = detail::init_bsf(packets->codecpar);
  if (!bsf_ctx) {
    return packets;
  }

  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      packets->src, bsf_ctx->par_out, packets->time_base);
  ret->timestamp = packets->timestamp;
  ret->frame_rate = packets->frame_rate;
  for (auto& packet : packets->get_packets()) {
    auto filtered = detail::apply_bsf(bsf_ctx.get(), packet);
    ret->push(filtered.release());
  }
  return ret;
}

} // namespace spdl::core
