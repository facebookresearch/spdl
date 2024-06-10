#include "libspdl/core/detail/ffmpeg/decoder.h"

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

namespace spdl::core::detail {
namespace {
void send_packet(AVCodecContext* codec_ctx, AVPacket* packet) {
  {
    TRACE_EVENT("decoding", "avcodec_send_packet");
    CHECK_AVERROR(
        avcodec_send_packet(codec_ctx, packet),
        "Failed to pass a frame to decoder.");
  }
}

int receive_frame(AVCodecContext* codec_ctx, AVFrame* frame) {
  int ret;
  {
    TRACE_EVENT("decoding", "avcodec_receive_frame");
    ret = avcodec_receive_frame(codec_ctx, frame);
  }
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to decode a packet.");
  }
  return ret;
}
} // namespace

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

Decoder::Decoder(
    AVCodecParameters* codecpar,
    Rational time_base,
    const std::optional<DecodeConfig>& cfg)
    : codec_ctx(detail::get_decode_codec_ctx_ptr(
          codecpar,
          time_base,
          cfg ? cfg->decoder : std::nullopt,
          cfg ? cfg->decoder_options : std::nullopt)) {}

Generator<AVFrame*> Decoder::decode(AVPacket* packet, bool flush_null) {
  VLOG(9)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  auto frame = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};
  send_packet(codec_ctx.get(), packet);
  int errnum;
  do {
    switch ((errnum = receive_frame(codec_ctx.get(), frame.get()))) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF: {
        if (flush_null) {
          co_yield nullptr;
        }
        co_return;
      }
      default: {
        {
          double ts = TS(frame, codec_ctx->pkt_timebase);
          VLOG(9) << fmt::format(
              "{:21s} {:.3f} ({})", " --- raw frame:", ts, frame->pts);
        }
        co_yield frame.get();
      }
    }
  } while (errnum >= 0);
}

} // namespace spdl::core::detail
