#include <libspdl/detail/demuxer.h>
#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/detail/ffmpeg/wrappers.h>
#include <libspdl/interface.h>
#include <libspdl/logging.h>

extern "C" {
#include <libavformat/avformat.h>
}

namespace spdl::detail {
namespace {
inline int parse_fmt_ctx(AVFormatContext* fmt_ctx, enum AVMediaType type) {
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      "Failed to find stream information.");

  int idx = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
  if (idx < 0) {
    SPDL_FAIL(
        fmt::format("No {} stream was found.", av_get_media_type_string(type)));
  }
  return idx;
}
} // namespace

folly::coro::AsyncGenerator<PackagedAVPackets&&> stream_demux(
    const enum AVMediaType type,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const IOConfig& cfg) {
  auto dp =
      get_data_provider(src, cfg.format, cfg.format_options, cfg.buffer_size);
  AVFormatContext* fmt_ctx = dp->get_fmt_ctx();
  int idx = parse_fmt_ctx(fmt_ctx, type);
  AVStream* stream = fmt_ctx->streams[idx];

  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }

  for (auto& timestamp : timestamps) {
    auto [start, end] = timestamp;

    int64_t t = static_cast<int64_t>(start * AV_TIME_BASE);
    CHECK_AVERROR(
        av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
        "Failed to seek to the timestamp {} [sec]",
        start);

    PackagedAVPackets package{
        fmt_ctx->url,
        timestamp,
        stream->codecpar,
        stream->time_base,
        av_guess_frame_rate(fmt_ctx, stream, nullptr)};
    double packet_ts = 0;
    do {
      AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
      int errnum = av_read_frame(fmt_ctx, packet.get());
      if (errnum == AVERROR_EOF) {
        break;
      }
      CHECK_AVERROR_NUM(errnum, "Failed to process packet.");
      if (packet->stream_index != idx) {
        continue;
      }
      packet_ts = packet->pts * av_q2d(stream->time_base);
      package.packets.push_back(packet.release());
      // Note:
      // Since the video frames can be non-chronological order, so we add small
      // margin to end
    } while (packet_ts < end + 0.3);
    XLOG(DBG) << fmt::format(" - Sliced {} packets", package.packets.size());
    // For flushing
    package.packets.push_back(nullptr);
    co_yield std::move(package);
  }
}

} // namespace spdl::detail
