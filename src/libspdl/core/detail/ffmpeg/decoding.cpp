#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace spdl::core {
namespace detail {
////////////////////////////////////////////////////////////////////////////////
// decoding functions
////////////////////////////////////////////////////////////////////////////////

namespace {
Generator<AVFramePtr> decode_packets(
    const std::vector<AVPacket*>& packets,
    Decoder& decoder,
    FilterGraph& filter) {
  for (auto& packet : packets) {
    auto decoding = decoder.decode(packet, !packet);
    while (decoding) {
      auto filtering = filter.filter(decoding());
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
