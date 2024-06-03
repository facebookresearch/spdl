#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace spdl::core {
namespace detail {

////////////////////////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////////////////////////

IterativeDecoding::Ite::Ite(
    Decoder* decoder_,
    AVPacket* packet,
    bool flush_null)
    : decoder(decoder_), null_flushed(!flush_null) {
  decoder->add_packet(packet);
  fill_next();
}

IterativeDecoding::Ite& IterativeDecoding::Ite::operator++() {
  fill_next();
  return *this;
}

bool IterativeDecoding::Ite::operator!=(const Sentinel&) {
  return !(completed && null_flushed);
}

AVFramePtr IterativeDecoding::Ite::operator*() {
  if (completed) {
    if (null_flushed) {
      // This should not be reachable, because `operator!=` guards
      SPDL_FAIL_INTERNAL("Frame was requested but decoding is exhausted.");
    }
    null_flushed = true;
    return {};
  }
  return std::move(next_ret);
}

void IterativeDecoding::Ite::fill_next() {
  next_ret = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};
  int errnum = decoder->get_frame(next_ret.get());
  switch (errnum) {
    case AVERROR(EAGAIN):
    case AVERROR_EOF: {
      completed = true;
      // Note: `next_ret` is now invalid state.
    }
  }
}

IterativeDecoding::IterativeDecoding(
    Decoder* decoder_,
    AVPacket* packet_,
    bool flush_null_)
    : decoder(decoder_), packet(packet_), flush_null(flush_null_){};

IterativeDecoding::Ite IterativeDecoding::begin() {
  return {decoder, packet, flush_null};
};

const IterativeDecoding::Sentinel& IterativeDecoding::end() {
  return sentinel;
};

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
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

void Decoder::add_packet(AVPacket* packet) {
  XLOG(DBG9)
      << ((!packet) ? fmt::format(" -- flush decoder")
                    : fmt::format(
                          "{:21s} {:.3f} ({})",
                          " -- packet:",
                          TS(packet, codec_ctx->pkt_timebase),
                          packet->pts));

  {
    TRACE_EVENT("decoding", "avcodec_send_packet");
    CHECK_AVERROR(
        avcodec_send_packet(codec_ctx.get(), packet),
        "Failed to pass a frame to decoder.");
  }
}

int Decoder::get_frame(AVFrame* frame) {
  int ret;
  {
    TRACE_EVENT("decoding", "avcodec_receive_frame");
    ret = avcodec_receive_frame(codec_ctx.get(), frame);
  }
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to decode a packet.");
  }
  if (ret >= 0) {
    double ts = TS(frame, codec_ctx->pkt_timebase);
    XLOG(DBG9) << fmt::format(
        "{:21s} {:.3f} ({})", " --- raw frame:", ts, frame->pts);
  }

  return ret;
}

IterativeDecoding Decoder::decode(AVPacket* packet, bool flush_null) {
  return {this, packet, flush_null};
}

////////////////////////////////////////////////////////////////////////////////
// decoding functions
////////////////////////////////////////////////////////////////////////////////

namespace {
Generator<AVFramePtr> decode_packets(
    const std::vector<AVPacket*>& packets,
    Decoder& decoder,
    FilterGraph& filter) {
  for (auto& packet : packets) {
    for (auto raw_frame : decoder.decode(packet, !packet)) {
      auto filtering = filter.filter(raw_frame.get());
      while (filtering) {
        co_yield filtering();
      }
    }
  }
}

template <MediaType media_type>
FilterGraph get_filter(
    AVCodecContext* codec_ctx,
    const std::string& filter_desc,
    std::optional<Rational> frame_rate) {
  if constexpr (media_type == MediaType::Audio) {
    return get_audio_filter(filter_desc, codec_ctx);
  }
  if constexpr (media_type == MediaType::Video) {
    return get_video_filter(filter_desc, codec_ctx, *frame_rate);
  }
  if constexpr (media_type == MediaType::Image) {
    return get_image_filter(filter_desc, codec_ctx);
  }
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> get_frame(DemuxedPackets<media_type>* packets) {
  return std::make_unique<FFmpegFrames<media_type>>(
      packets->id, packets->time_base);
}

} // namespace
} // namespace detail

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc) {
  TRACE_EVENT(
      "decoding",
      "decode_packets_ffmpeg",
      perfetto::Flow::ProcessScoped(packets->id));
  detail::Decoder decoder{packets->codecpar, packets->time_base, cfg};
  if constexpr (media_type != MediaType::Image) {
    packets->push(nullptr); // For flushing
  }
  auto filter = detail::get_filter<media_type>(
      decoder.codec_ctx.get(), filter_desc, packets->frame_rate);
  auto ret = detail::get_frame(packets.get());

  auto gen = detail::decode_packets(packets->get_packets(), decoder, filter);
  while (gen) {
    ret->push_back(gen().release());
  }
  ret->time_base = filter.get_sink_time_base();
  return ret;
}

template FFmpegAudioFramesPtr decode_packets_ffmpeg(
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template FFmpegVideoFramesPtr decode_packets_ffmpeg(
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

template FFmpegImageFramesPtr decode_packets_ffmpeg(
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig> cfg,
    std::string filter_desc);

////////////////////////////////////////////////////////////////////////////////
// StreamingDecoder
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
  requires(media_type != MediaType::Image)
StreamingDecoder<media_type>::Impl::Impl(
    PacketsPtr<media_type> packets_,
    const std::optional<DecodeConfig> cfg_,
    const std::string filter_desc_)
    : packets(std::move(packets_)),
      decoder(packets->codecpar, packets->time_base, cfg_),
      filter_graph(detail::get_filter<media_type>(
          decoder.codec_ctx.get(),
          filter_desc_,
          packets->frame_rate)),
      gen(detail::decode_packets(
          packets->get_packets(),
          decoder,
          filter_graph)) {
  packets->push(nullptr);
}

template <MediaType media_type>
  requires(media_type != MediaType::Image)
std::optional<FFmpegFramesPtr<media_type>>
StreamingDecoder<media_type>::Impl::decode(int num_frames) {
  if (num_frames <= 0) {
    SPDL_FAIL("the `num_frames` must be positive.");
  }

  if (!gen) {
    return {};
  }

  TRACE_EVENT("decoding", "StreamingDecoder::decode");
  auto ret = detail::get_frame(packets.get());
  for (int i = 0; gen && (i < num_frames); ++i) {
    ret->push_back(gen().release());
  }
  return ret;
}

template struct StreamingDecoder<MediaType::Video>::Impl;

} // namespace spdl::core
