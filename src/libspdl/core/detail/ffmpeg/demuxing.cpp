#include "libspdl/core/detail/ffmpeg/demuxing.h"

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

namespace spdl::core::detail {
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

std::unique_ptr<DataInterface> get_interface(
    std::string_view src,
    SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg) {
  if (!adaptor) {
    thread_local auto p = std::make_shared<SourceAdaptor>();
    adaptor = p;
  }
  return adaptor->get(src, io_cfg);
}

folly::coro::AsyncGenerator<AVPacketPtr> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::tuple<double, double>& timestamp) {
  TRACE_EVENT("demuxing", "detail::demux_window");
  auto [start, end] = timestamp;

  if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    // Note:
    // Since the video frames can be non-chronological order, so we add small
    // margin to end
    end += 0.3;
  }

  int64_t t = static_cast<int64_t>(start * AV_TIME_BASE);
  {
    TRACE_EVENT("demuxing", "av_seek_frame");
    CHECK_AVERROR(
        av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
        "Failed to seek to the timestamp {} [sec]",
        start);
  }

  double packet_ts = -1;
  do {
    co_await folly::coro::co_safe_point;
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
      co_yield std::move(packet);
    }
  } while (packet_ts < end);
}

} // namespace

template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux(
    std::string_view src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    const IOConfig io_cfg) {
  TRACE_EVENT("demuxing", "detail::stream_demux");
  auto interface = get_interface(src, adaptor, io_cfg);
  AVFormatContext* fmt_ctx = interface->get_fmt_ctx();
  AVStream* stream = init_fmt_ctx(fmt_ctx, media_type);
  auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);

  for (auto& timestamp : timestamps) {
    co_await folly::coro::co_safe_point;
    auto package = std::make_unique<DemuxedPackets<media_type>>(
        fmt_ctx->url,
        timestamp,
        stream->codecpar,
        Rational{stream->time_base.num, stream->time_base.den},
        Rational{frame_rate.num, frame_rate.den});
    auto task = demux_window(fmt_ctx, stream, timestamp);
    while (auto packet = co_await task.next()) {
      package->push(packet->release());
    }
    XLOG(DBG9) << fmt::format(" - Sliced {} packets", package->num_packets());
    co_yield std::move(package);
  }
}

template folly::coro::AsyncGenerator<AudioPacketsPtr>
stream_demux<MediaType::Audio>(
    std::string_view src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    const IOConfig io_cfg);

template folly::coro::AsyncGenerator<VideoPacketsPtr>
stream_demux<MediaType::Video>(
    std::string_view src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    const IOConfig io_cfg);

folly::coro::Task<ImagePacketsPtr> demux_image(
    std::string_view src,
    SourceAdaptorPtr adaptor,
    const IOConfig io_cfg) {
  TRACE_EVENT("demuxing", "detail::demux");
  auto interface = get_interface(src, adaptor, io_cfg);
  AVFormatContext* fmt_ctx = interface->get_fmt_ctx();
  AVStream* stream = init_fmt_ctx(fmt_ctx, MediaType::Video);

  auto package = std::make_unique<DemuxedPackets<MediaType::Image>>(
      fmt_ctx->url,
      std::tuple<double, double>{0., 1000.},
      stream->codecpar,
      Rational{1, 1},
      Rational{1, 1});

  int ite = 0;
  do {
    co_await folly::coro::co_safe_point;
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
  co_return std::move(package);
}

////////////////////////////////////////////////////////////////////////////////
// Bit Stream Filtering for NVDEC
////////////////////////////////////////////////////////////////////////////////
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

folly::coro::Task<VideoPacketsPtr> apply_bsf(VideoPacketsPtr packets) {
  TRACE_EVENT("demuxing", "apply_bsf");
  AVBSFContextPtr bsf_ctx = init_bsf(packets->codecpar);
  if (!bsf_ctx) {
    co_return packets;
  }

  auto package = std::make_unique<DemuxedPackets<MediaType::Video>>(
      packets->src,
      packets->timestamp,
      bsf_ctx->par_out,
      packets->time_base,
      packets->frame_rate);
  for (auto& packet : packets->get_packets()) {
    co_await folly::coro::co_safe_point;
    AVPacketPtr filtered = apply_bsf(bsf_ctx.get(), packet);
    package->push(filtered.release());
  }
  co_return package;
}

} // namespace spdl::core::detail
