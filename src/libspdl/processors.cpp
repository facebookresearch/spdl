#include <fmt/core.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/BoundedQueue.h>

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
//////////////////////////////////////////////////////////////////////////////
// PackagedAVFrames
//////////////////////////////////////////////////////////////////////////////
PackagedAVFrames::~PackagedAVFrames() {
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

namespace {

//////////////////////////////////////////////////////////////////////////////
// PackagedAVPackets
//////////////////////////////////////////////////////////////////////////////
// Struct passed from IO thread pool to decoder thread pool.
// Similar to PackagedAVFrames, AVFrame pointers are bulk released.
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

  PackagedAVFrames frames;
  frames.time_base = to_tuple(
      filter_graph ? filter_graph->filters[1]->inputs[0]->time_base
                   : packets.time_base);
  auto decoding = filter_graph
      ? decode_packets(packets.packets, codec_ctx.get(), filter_graph.get())
      : decode_packets(packets.packets, codec_ctx.get());
  while (auto frame = co_await decoding.next()) {
    frames.frames.push_back(*frame);
  }
  XLOG(DBG) << fmt::format("Generated {} frames", frames.frames.size());
  co_await queue.enqueue(std::move(frames));
}

Task<void> stream_decode(
    VideoDecodingJob job,
    folly::Executor::KeepAlive<> decode_exec,
    FrameQueue& queue) {
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
                         .scheduleOn(decode_exec)
                         .start());
  }
  co_await folly::collectAll(sfs);
}
} // namespace

//////////////////////////////////////////////////////////////////////////////
// Engine
//////////////////////////////////////////////////////////////////////////////
Engine::Engine(
    size_t num_io_threads,
    size_t num_decoding_threads,
    size_t frame_queue_size)
    : io_task_executors(std::make_unique<Executor>(num_io_threads)),
      decoding_task_executors(std::make_unique<Executor>(num_decoding_threads)),
      io_exec(io_task_executors.get()),
      decoding_exec(decoding_task_executors.get()),
      frame_queue(frame_queue_size) {}

void Engine::enqueue(VideoDecodingJob job) {
  sfs.emplace_back(stream_decode(std::move(job), decoding_exec, frame_queue)
                       .scheduleOn(io_exec)
                       .start());
}

VideoBuffer convert_rgb24(PackagedAVFrames& val) {
  assert(val.frames[0]->format == AV_PIX_FMT_RGB24);
  VideoBuffer buf;
  buf.n = val.frames.size();
  buf.c = 3;
  buf.h = static_cast<size_t>(val.frames[0]->height);
  buf.w = static_cast<size_t>(val.frames[0]->width);
  buf.channel_last = true;
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  size_t linesize = buf.c * buf.w;
  uint8_t* dst = buf.data.data();
  for (const auto& frame : val.frames) {
    uint8_t* src = frame->data[0];
    for (int i = 0; i < frame->height; ++i) {
      memcpy(dst, src, linesize);
      src += frame->linesize[0];
      dst += linesize;
    }

    XLOG(DBG) << fmt::format(
        " - {:.3f} ({}x{}, {})",
        ts(frame->pts, val.time_base),
        frame->width,
        frame->height,
        av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
  }
  return buf;
}

VideoBuffer convert_yuv420p(PackagedAVFrames& val) {
  assert(val.frames[0]->format == AV_PIX_FMT_YUV420P);
  size_t height = val.frames[0]->height;
  size_t width = val.frames[0]->width;
  assert(height % 2 == 0 && width % 2 == 0);
  size_t h2 = height / 2;
  size_t w2 = width / 2;
  VideoBuffer buf;
  buf.n = val.frames.size();
  buf.c = 1;
  buf.h = height + h2;
  buf.w = width;
  buf.channel_last = false;
  buf.data.resize(buf.n * buf.h * buf.w);

  uint8_t* dst = buf.data.data();
  for (const auto& frame : val.frames) {
    // Y
    {
      uint8_t* src = frame->data[0];
      size_t linesize = buf.w;
      for (int i = 0; i < frame->height; ++i) {
        memcpy(dst, src, linesize);
        src += frame->linesize[0];
        dst += linesize;
      }
    }
    // U & V
    {
      uint8_t* src_u = frame->data[1];
      uint8_t* src_v = frame->data[2];
      size_t linesize = w2;
      for (int i = 0; i < h2; ++i) {
        // U
        memcpy(dst, src_u, linesize);
        src_u += frame->linesize[1];
        dst += linesize;
        // V
        memcpy(dst, src_v, linesize);
        src_v += frame->linesize[2];
        dst += linesize;
      }
    }
    XLOG(DBG) << fmt::format(
        " - {:.3f} ({}x{}, {})",
        ts(frame->pts, val.time_base),
        frame->width,
        frame->height,
        av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
  }
  return buf;
}

VideoBuffer Engine::dequeue() {
  VideoBuffer val =
      folly::coro::blockingWait([&]() -> folly::coro::Task<VideoBuffer> {
        auto val = co_await frame_queue.dequeue();
        // TODO: validate the number of frames > 0
        XLOG(DBG) << fmt::format("Dequeue {} frames", val.frames.size());
        switch (static_cast<AVPixelFormat>(val.frames[0]->format)) {
          case AV_PIX_FMT_RGB24:
            co_return convert_rgb24(val);
          case AV_PIX_FMT_YUV420P:
            co_return convert_yuv420p(val);
          default:
            throw std::runtime_error(fmt::format(
                "Unsupported pixel format: {}",
                av_get_pix_fmt_name(
                    static_cast<AVPixelFormat>(val.frames[0]->format))));
        }
      }());

  XLOG(DBG) << fmt::format(
      "Buffer returned: {}x{}x{}x{} ({})",
      val.n,
      val.c,
      val.h,
      val.w,
      fmt::ptr(val.data.data()));
  return val;
}

} // namespace spdl
