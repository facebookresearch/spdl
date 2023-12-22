#include <fmt/core.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/BoundedQueue.h>

#include <libspdl/detail/executors.h>
#include <libspdl/detail/ffmpeg/ctx_utils.h>
#include <libspdl/detail/ffmpeg/filter_graph.h>
#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/interface.h>
#include <libspdl/processors.h>
#include <cstddef>
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

using namespace spdl::detail;

template <typename T>
using Task = folly::coro::Task<T>;

template <typename T>
using Generator = folly::coro::AsyncGenerator<T>;

namespace spdl {
namespace {

//////////////////////////////////////////////////////////////////////////////
// PackagedAVPackets
//////////////////////////////////////////////////////////////////////////////
// Struct passed from IO thread pool to decoder thread pool.
// Similar to Frames, AVFrame pointers are bulk released.
// It contains sufficient information to build decoder via AVStream*.
struct PackagedAVPackets {
  // frame rate for video
  AVRational frame_rate;

  // Requested timestamp
  std::string src;
  double timestamp;

  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

  PackagedAVPackets(AVRational frame_rate, std::string src, double timestamp)
      : frame_rate(frame_rate), src(src), timestamp(timestamp){};

  // No copy constructors
  PackagedAVPackets(const PackagedAVPackets&) = delete;
  PackagedAVPackets& operator=(const PackagedAVPackets&) = delete;
  // Move constructor to support AsyncGenerator
  PackagedAVPackets(PackagedAVPackets&&) noexcept = default;
  PackagedAVPackets& operator=(PackagedAVPackets&&) noexcept = default;
  // Destructor releases AVPacket* resources
  ~PackagedAVPackets() {
    std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
      if (p) {
        av_packet_unref(p);
        av_packet_free(&p);
      }
    });
  };
};

//////////////////////////////////////////////////////////////////////////////
// Demuxer
//////////////////////////////////////////////////////////////////////////////
int parse_fmt_ctx(AVFormatContext* fmt_ctx) {
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      "Failed to find stream information.");

  int idx =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (idx < 0) {
    throw std::runtime_error("No video stream was found.");
  }
  return idx;
}

Generator<PackagedAVPackets&&> streaming_demux(
    AVFormatContext* fmt_ctx,
    int idx,
    std::vector<double> timestamps) {
  AVStream* stream = fmt_ctx->streams[idx];

  // Disable other streams
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (i != idx) {
      fmt_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }

  for (auto& timestamp : timestamps) {
    int64_t t = static_cast<int64_t>(timestamp * AV_TIME_BASE);
    CHECK_AVERROR(
        av_seek_frame(fmt_ctx, -1, t, AVSEEK_FLAG_BACKWARD),
        "Failed to seek to the timestamp {} [sec]",
        timestamp);

    int num_req_frames = 10;

    PackagedAVPackets package{
        av_guess_frame_rate(fmt_ctx, stream, nullptr), fmt_ctx->url, timestamp};
    while (num_req_frames >= 0) {
      AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
      int errnum = av_read_frame(fmt_ctx, packet.get());
      if (errnum == AVERROR_EOF) {
        break;
      }
      CHECK_AVERROR_NUM(errnum, "Failed to process packet.");
      if (packet->stream_index != idx) {
        continue;
      }
      // Note:
      // Since the video frames can be non-chronological order, this is not
      // correct.
      // TODO: fix this.
      double packet_ts = packet->pts * av_q2d(stream->time_base);
      if (packet_ts > timestamp) {
        num_req_frames -= 1;
      }
      package.packets.push_back(packet.release());
    }
    XLOG(DBG) << fmt::format(" - Sliced {} packets", package.packets.size());
    // For flushing
    package.packets.push_back(nullptr);
    co_yield std::move(package);
  }
}

//////////////////////////////////////////////////////////////////////////////
// Decoder and filter
//////////////////////////////////////////////////////////////////////////////

#define TS(PTS, BASE) (static_cast<double>(PTS) * BASE.num / BASE.den)

Generator<AVFrame*> filter_frame(AVFrame* frame, AVFilterGraph* filter_graph) {
  assert(filter_graph);
  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];

  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  XLOG(DBG)
      << ((!frame) ? fmt::format(" --- flush filter graph")
                   : fmt::format(
                         "{:15s} {:.3f} ({})",
                         " --- raw frame:",
                         TS(frame->pts, src_ctx->outputs[0]->time_base),
                         frame->pts));

  int errnum =
      av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);

  CHECK_AVERROR_NUM(errnum, "Failed to pass a frame to filter.");

  AVFrameAutoUnref frame_ref{frame};
  while (errnum >= 0) {
    AVFramePtr frame2{CHECK_AVALLOCATE(av_frame_alloc())};
    errnum = av_buffersink_get_frame(sink_ctx, frame2.get());
    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_return;
      default: {
        CHECK_AVERROR_NUM(errnum, "Failed to filter a frame.");

        XLOG(DBG) << fmt::format(
            "{:15s} {:.3f} ({})",
            " ---- frame:",
            TS(frame2->pts, sink_ctx->inputs[0]->time_base),
            frame2->pts);

        co_yield frame2.release();
      }
    }
  }
}

Generator<AVFrame*> decode_packet(AVCodecContext* codec_ctx, AVPacket* packet) {
  assert(codec_ctx);
  XLOG(DBG)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:15s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet->pts, codec_ctx->pkt_timebase),
                          packet->pts));

  int errnum = avcodec_send_packet(codec_ctx, packet);
  while (errnum >= 0) {
    AVFramePtr frame{CHECK_AVALLOCATE(av_frame_alloc())};
    errnum = avcodec_receive_frame(codec_ctx, frame.get());
    switch (errnum) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_yield nullptr;
        co_return;
      default: {
        CHECK_AVERROR_NUM(errnum, "Failed to decode a frame.");
        co_yield frame.release();
      }
    }
  }
}

Task<Frames> decode_packets(
    PackagedAVPackets packets,
    AVCodecContextPtr codec_ctx) {
  Frames frames;
  for (auto& packet : packets.packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto frame = co_await decoding.next()) {
      if (*frame) {
        XLOG(DBG) << fmt::format(
            "{:15s} {:.3f} ({})",
            " --- raw frame:",
            TS((*frame)->pts, codec_ctx->pkt_timebase),
            (*frame)->pts);

        frames.frames.push_back(*frame);
      }
    }
  }
  co_return frames;
}

Task<Frames> decode_packets(
    PackagedAVPackets packets,
    AVCodecContextPtr codec_ctx,
    const std::string filter_desc) {
  auto filter_graph =
      get_video_filter(filter_desc, codec_ctx.get(), packets.frame_rate);

  // XLOG(DBG) << describe_graph(filter_graph.get());

  // Note:
  // The time_base of filtered frames can be found at
  // `filter_graph->filters[1]->inputs[0]->time_base`
  Frames frames;
  for (auto& packet : packets.packets) {
    auto decoding = decode_packet(codec_ctx.get(), packet);
    while (auto raw_frame = co_await decoding.next()) {
      AVFramePtr f{*raw_frame};
      auto filtering = filter_frame(*raw_frame, filter_graph.get());
      while (auto frame = co_await filtering.next()) {
        frames.frames.push_back(*frame);
      }
    }
  }
  co_return frames;
}

Task<void> process_packets(
    PackagedAVPackets packets,
    AVCodecContextPtr codec_ctx,
    const std::string filter_desc,
    FrameQueue& queue) {
  auto decode_job = filter_desc.empty()
      ? decode_packets(std::move(packets), std::move(codec_ctx))
      : decode_packets(std::move(packets), std::move(codec_ctx), filter_desc);
  co_await queue.enqueue(co_await std::move(decode_job));
}

Task<void> stream_decode(VideoDecodingJob job, FrameQueue& q) {
  auto dp = get_data_provider(
      job.src, job.format, job.format_options, job.buffer_size);

  auto fmt_ctx = dp->get_fmt_ctx();
  int idx = parse_fmt_ctx(fmt_ctx);
  AVStream* stream = fmt_ctx->streams[idx];

  auto filter_desc = get_video_filter_description(
      job.frame_rate,
      job.width,
      job.height,
      job.pix_fmt,
      static_cast<AVPixelFormat>(stream->codecpar->format));

  auto demuxer = streaming_demux(fmt_ctx, idx, job.timestamps);
  std::vector<folly::SemiFuture<folly::Unit>> sfs;
  auto exec = getDecoderThreadPoolExecutor();
  while (auto packets = co_await demuxer.next()) {
    auto codec_ctx = get_codec_ctx(
        stream->codecpar,
        stream->time_base,
        job.decoder,
        job.decoder_options,
        job.cuda_device_index);

    sfs.emplace_back(
        process_packets(*packets, std::move(codec_ctx), filter_desc, q)
            .scheduleOn(exec)
            .start());
  }
  co_await folly::collectAll(sfs);
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// Engine
//////////////////////////////////////////////////////////////////////////////
Engine::Engine(size_t frame_queue_size) : frame_queue(frame_queue_size) {}

void Engine::enqueue(VideoDecodingJob job) {
  auto exec = getDemuxerThreadPoolExecutor();
  sfs.emplace_back(
      stream_decode(std::move(job), frame_queue).scheduleOn(exec).start());
}

Frames Engine::dequeue() {
  Frames frames = folly::coro::blockingWait(frame_queue.dequeue());
  // TODO: validate the number of frames > 0
  XLOG(DBG) << fmt::format("Dequeue {} frames", frames.frames.size());
  return frames;
}

} // namespace spdl
