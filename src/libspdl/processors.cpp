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
  // Owned by the top level AVFormatContext that client code keeps.
  AVCodecParameters* codec_par = nullptr;
  // Packet time base
  AVRational time_base{};
  // frame rate for video
  AVRational frame_rate;

  // Requested timestamp
  std::string src;
  double timestamp = -1.;

  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

  PackagedAVPackets(
      AVStream* stream,
      AVRational frame_rate,
      std::string src,
      double timestamp)
      : codec_par(stream->codecpar),
        time_base(stream->time_base),
        frame_rate(frame_rate),
        src(src),
        timestamp(timestamp){};

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
Generator<PackagedAVPackets&&> streaming_demux(
    AVFormatContext* fmt_ctx,
    std::vector<double> timestamps) {
  CHECK_AVERROR(
      avformat_find_stream_info(fmt_ctx, nullptr),
      "Failed to find stream information.");

  int idx =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (idx < 0) {
    throw std::runtime_error("No video stream was found.");
  }
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
        stream,
        av_guess_frame_rate(fmt_ctx, stream, nullptr),
        fmt_ctx->url,
        timestamp};
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
double ts(int64_t pts, Rational time_base) {
  auto& [num, den] = time_base;
  return static_cast<double>(pts) * num / den;
}

inline Rational to_tuple(const AVRational& r) {
  return {r.num, r.den};
}

Generator<AVFrame*> filter_frame(AVFrame* frame, AVFilterGraph* filter_graph) {
  assert(filter_graph);
  AVFilterContext* src_ctx = filter_graph->filters[0];
  AVFilterContext* sink_ctx = filter_graph->filters[1];

  assert(strcmp(src_ctx->name, "in") == 0);
  assert(strcmp(sink_ctx->name, "out") == 0);

  // XLOG(DBG)
  //     << ((!frame) ? fmt::format(" --- flush filter graph")
  //                  : fmt::format(
  //                        "{:15s} {:.3f} ({})",
  //                        " --- raw frame:",
  //                        ts(frame->pts, src_ctx->outputs[0]->time_base),
  //                        frame->pts));

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

        // XLOG(DBG) << fmt::format(
        //     "{:15s} {:.3f} ({})",
        //     " ---- frame:",
        //     ts(frame2->pts, sink_ctx->inputs[0]->time_base),
        //     frame2->pts);
        co_yield frame2.release();
      }
    }
  }
}

Generator<AVFrame*> decode_packet(AVCodecContext* codec_ctx, AVPacket* packet) {
  assert(codec_ctx);
  // XLOG(DBG)
  //     << ((!packet) ? fmt::format(" -- flush decoder")
  //                   : fmt::format(
  //                         "{:15s} {:.3f} ({})",
  //                         " -- packet:",
  //                         ts(packet->pts, codec_ctx->pkt_timebase),
  //                         packet->pts));

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

Generator<AVFrame*> decode_packets(
    const std::vector<AVPacket*>& packets,
    AVCodecContext* codec_ctx) {
  for (auto& packet : packets) {
    auto decoding = decode_packet(codec_ctx, packet);
    while (auto raw_frame = co_await decoding.next()) {
      if (AVFrame* _f = *raw_frame; _f) {
        co_yield _f;
      }
    }
  }
}

Generator<AVFrame*> decode_packets(
    const std::vector<AVPacket*>& packets,
    AVCodecContext* codec_ctx,
    AVFilterGraph* filter_graph) {
  for (auto& packet : packets) {
    auto decoding = decode_packet(codec_ctx, packet);
    while (auto raw_frame = co_await decoding.next()) {
      AVFramePtr f{*raw_frame};
      auto filtering = filter_frame(*raw_frame, filter_graph);
      while (auto frame = co_await filtering.next()) {
        co_yield *frame;
      }
    }
  }
}

Task<void> decode_and_enque(
    PackagedAVPackets packets,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_options,
    const int cuda_device_index,
    const std::optional<Rational>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt,
    FrameQueue& queue) {
  auto codec_ctx = get_codec_ctx(
      packets.codec_par,
      packets.time_base,
      decoder,
      decoder_options,
      cuda_device_index);
  auto filter_graph = [&]() {
    auto filter_desc =
        get_video_filter_description(frame_rate, width, height, pix_fmt);
    if (filter_desc.empty()) {
      return AVFilterGraphPtr{nullptr};
    }
    return get_video_filter(
        filter_desc, codec_ctx.get(), packets.time_base, packets.frame_rate);
  }();

  // XLOG(DBG) << describe_graph(filter_graph.get());

  Frames frames;
  // Note:
  // Way to retrieve the time base of the decoded frame are different
  // whether filter graph is used or not
  // with filter graph, it's `filter_graph->filters[1]->inputs[0]->time_base`
  // otherwise packets.time_base
  auto decoding = filter_graph
      ? decode_packets(packets.packets, codec_ctx.get(), filter_graph.get())
      : decode_packets(packets.packets, codec_ctx.get());
  while (auto frame = co_await decoding.next()) {
    frames.frames.push_back(*frame);
  }
  XLOG(DBG) << fmt::format("Generated {} frames", frames.frames.size());
  co_await queue.enqueue(std::move(frames));
}

Task<void> stream_decode(VideoDecodingJob job, FrameQueue& queue) {
  auto dp = get_data_provider(
      job.src, job.format, job.format_options, job.buffer_size);
  auto demuxer = streaming_demux(dp->get_fmt_ctx(), job.timestamps);
  std::vector<folly::SemiFuture<folly::Unit>> sfs;
  while (auto packets = co_await demuxer.next()) {
    sfs.emplace_back(decode_and_enque(
                         *packets,
                         job.decoder,
                         job.decoder_options,
                         job.cuda_device_index,
                         job.frame_rate,
                         job.width,
                         job.height,
                         job.pix_fmt,
                         queue)
                         .scheduleOn(getDecoderThreadPoolExecutor())
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
  sfs.emplace_back(stream_decode(std::move(job), frame_queue)
                       .scheduleOn(getDemuxerThreadPoolExecutor())
                       .start());
}

Frames Engine::dequeue() {
  Frames frames = folly::coro::blockingWait(frame_queue.dequeue());
  // TODO: validate the number of frames > 0
  XLOG(DBG) << fmt::format("Dequeue {} frames", frames.frames.size());
  return frames;
}

} // namespace spdl
