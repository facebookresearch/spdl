#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

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

int read_packet(AVFormatContext* fmt_ctx, AVPacket* packet) {
  int ret;
  {
    TRACE_EVENT("demuxing", "av_read_frame");
    ret = av_read_frame(fmt_ctx, packet);
  }
  if (ret < 0 && ret != AVERROR_EOF) {
    CHECK_AVERROR_NUM(
        ret, fmt::format("Failed to read a packet. ({})", fmt_ctx->url));
  }
  return ret;
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

  do {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    if (read_packet(fmt_ctx, packet.get()) == AVERROR_EOF) {
      break;
    }
    if (packet->stream_index != stream->index) {
      continue;
    }
    if (packet->pts * av_q2d(stream->time_base) > end) {
      break;
    }
    ret->push(packet.release());
  } while (true);
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
Demuxer<media_type>::Demuxer(std::unique_ptr<DataInterface> di_)
    : di(std::move(di_)),
      fmt_ctx(di->get_fmt_ctx()),
      stream(detail::init_fmt_ctx(fmt_ctx, media_type)){};

template <MediaType media_type>
Demuxer<media_type>::~Demuxer() {
  TRACE_EVENT("demuxing", "Demuxer::~Demuxer");
  di.reset();
  // Techinically, this is not necessary, but doing it here puts
  // the destruction of AVFormatContext under ~StreamingDemuxe, which
  // makes the trace easier to interpret.
}

template <MediaType media_type>
PacketsPtr<media_type> Demuxer<media_type>::demux_window(
    const std::optional<std::tuple<double, double>>& window) {
  auto packets = detail::demux_window<media_type>(fmt_ctx, stream, window);
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    packets->frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return packets;
}

template class Demuxer<MediaType::Audio>;
template class Demuxer<MediaType::Video>;

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer<media_type>>(
      detail::get_interface(src, adaptor, dmx_cfg));
}

template DemuxerPtr<MediaType::Audio> make_demuxer(
    const std::string,
    const SourceAdaptorPtr&,
    const std::optional<DemuxConfig>&);
template DemuxerPtr<MediaType::Video> make_demuxer(
    const std::string,
    const SourceAdaptorPtr&,
    const std::optional<DemuxConfig>&);

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer<media_type>>(
      detail::get_in_memory_interface(data, dmx_cfg));
}

template DemuxerPtr<MediaType::Audio> make_demuxer(
    const std::string_view,
    const std::optional<DemuxConfig>&);
template DemuxerPtr<MediaType::Video> make_demuxer(
    const std::string_view,
    const std::optional<DemuxConfig>&);

////////////////////////////////////////////////////////////////////////////////
// Demuxing for Image
////////////////////////////////////////////////////////////////////////////////
namespace detail {
namespace {
ImagePacketsPtr demux_image(AVFormatContext* fmt_ctx) {
  TRACE_EVENT("demuxing", "detail::demux");
  AVStream* stream = init_fmt_ctx(fmt_ctx, MediaType::Video);

  auto ret = std::make_unique<DemuxedPackets<MediaType::Image>>(
      fmt_ctx->url, stream->codecpar, Rational{1, 1});

  do {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    if (read_packet(fmt_ctx, packet.get()) == AVERROR_EOF) {
      break;
    }
    if (packet->stream_index != stream->index) {
      continue;
    }
    ret->push(packet.release());
  } while (true);
  if (ret->num_packets() != 1) {
    SPDL_FAIL(fmt::format(
        "Error demuxing an image from {}. "
        "Expected exactly one packet but found {}",
        fmt_ctx->url,
        ret->num_packets()));
  }
  return ret;
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
    const std::optional<DemuxConfig>& dmx_cfg) {
  thread_local SourceAdaptorPtr adaptor{new BytesAdaptor()};
  auto interface = detail::get_interface(data, adaptor, dmx_cfg);
  auto result = detail::demux_image(interface->get_fmt_ctx());
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Bit Stream Filtering for NVDEC
////////////////////////////////////////////////////////////////////////////////
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets) {
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
  switch (packets->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      name = "h264_mp4toannexb";
      break;
    case AV_CODEC_ID_HEVC:
      name = "hevc_mp4toannexb";
      break;
    default:
      return packets;
  }

  TRACE_EVENT("demuxing", "apply_bsf");
  auto bsf = detail::BitStreamFilter{name, packets->codecpar};

  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      packets->src, bsf.get_output_codec_par(), packets->time_base);
  ret->timestamp = packets->timestamp;
  ret->frame_rate = packets->frame_rate;
  for (auto& packet : packets->get_packets()) {
    auto filtering = bsf.filter(packet);
    while (filtering) {
      ret->push(filtering().release());
    }
  }
  return ret;
}

} // namespace spdl::core
