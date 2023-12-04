
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/rational.h>
}

#include <fmt/core.h>
#include <fmt/std.h>

#include <libspdl/ffmpeg/logging.h>
#include <libspdl/ffmpeg/utils.h>
#include <libspdl/processors.h>

#include <folly/experimental/coro/BoundedQueue.h>

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
  AVStream* stream = nullptr;

  // Requested timestamp
  std::string src;
  double timestamp = -1.;

  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

  PackagedAVPackets(AVStream* stream, std::string src, double timestamp)
      : stream(stream), src(src), timestamp(timestamp){};

  // No copy constructors
  PackagedAVPackets(const PackagedAVPackets&) = delete;
  PackagedAVPackets& operator=(const PackagedAVPackets&) = delete;
  // Move constructor to support AsyncGenerator
  PackagedAVPackets(PackagedAVPackets&&) noexcept = default;
  PackagedAVPackets& operator=(PackagedAVPackets&&) noexcept = default;
  // Destructor releases AVPacket* resources
  ~PackagedAVPackets() {
    std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
      av_packet_unref(p);
      av_packet_free(&p);
    });
  };
};

AVPacketPtr alloc_packet() {
  AVPacket* p = av_packet_alloc();
  if (!p) {
    throw std::runtime_error("Failed to allocate AVPacket.");
  }
  return AVPacketPtr{p};
}

AVFramePtr alloc_frame() {
  AVFrame* p = av_frame_alloc();
  if (!p) {
    throw std::runtime_error("Failed to allocate AVFrame.");
  }
  return AVFramePtr{p};
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// Producer
//////////////////////////////////////////////////////////////////////////////

folly::coro::AsyncGenerator<PackagedAVPackets&&> stream_avframes(
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

    PackagedAVPackets package{stream, fmt_ctx->url, timestamp};
    while (num_req_frames >= 0) {
      AVPacketPtr packet = alloc_packet();
      int errnum = av_read_frame(fmt_ctx, packet.get());
      if (errnum == AVERROR_EOF) {
        break;
      }
      if (errnum < 0) {
        throw std::runtime_error(av_error(errnum, "Failed to process packet."));
      }
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
    co_yield std::move(package);
  }
}

//////////////////////////////////////////////////////////////////////////////
// Consumer
//////////////////////////////////////////////////////////////////////////////
folly::coro::Task<void> decode_packets(
    PackagedAVPackets message,
    FrameQueue& queue) {
  AVCodecContextPtr codec_ctx = get_codec_ctx(message.stream->codecpar);
  PackagedAVFrames frames;
  for (auto& packet : message.packets) {
    int errnum = avcodec_send_packet(codec_ctx.get(), packet);
    while (errnum >= 0) {
      AVFramePtr frame = alloc_frame();
      errnum = avcodec_receive_frame(codec_ctx.get(), frame.get());
      if (errnum == AVERROR(EAGAIN)) {
        break;
      }
      if (errnum == AVERROR_EOF) {
        // co_wait flush_buffer()
        break;
      }
      if (errnum < 0) {
        throw std::runtime_error(av_error(errnum, "Failed to decode a frame."));
      }
      frames.frames.push_back(frame.release());
    }
  }

  LOG(INFO) << fmt::format(
      "    [{}] Target timestamp: {} - Decoded frames: {}",
      std::this_thread::get_id(),
      message.timestamp,
      frames.frames.size());
  co_await queue.enqueue(std::move(frames));
}

//////////////////////////////////////////////////////////////////////////////
// Processor
//////////////////////////////////////////////////////////////////////////////
folly::coro::Task<void> stream_decode(
    AVFormatContext* fmt_ctx,
    const std::vector<double> timestamps,
    FrameQueue& queue) {
  auto exec = folly::getGlobalCPUExecutor().get();
  auto streamer = stream_avframes(fmt_ctx, timestamps);
  std::vector<folly::SemiFuture<folly::Unit>> sfs;
  while (auto res = co_await streamer.next()) {
    sfs.emplace_back(decode_packets(*res, queue).scheduleOn(exec).start());
  }
  co_await folly::collectAll(sfs);
}

} // namespace spdl
