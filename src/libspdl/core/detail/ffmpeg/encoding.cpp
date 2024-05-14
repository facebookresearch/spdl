#include "libspdl/core/detail/ffmpeg/encoding.h"
#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"

namespace spdl::core::detail {

Encoder::Encoder(
    AVFormatOutputContextPtr&& format_ctx_,
    AVStream* stream_,
    AVCodecContextPtr&& codec_ctx_)
    : format_ctx(std::move(format_ctx_)),
      stream(stream_),
      codec_ctx(std::move(codec_ctx_)),
      packet(CHECK_AVALLOCATE(av_packet_alloc())){};

void Encoder::encode(const AVFramePtr& frame) {
  auto fmt_ctx = format_ctx.get();
  auto cdc_ctx = codec_ctx.get();
  auto pkt = packet.get();

  CHECK_AVERROR(
      avcodec_send_frame(cdc_ctx, frame.get()), "Failed to encode frame.");

  int ret;
  do {
    ret = avcodec_receive_packet(cdc_ctx, pkt);
    if (ret == AVERROR_EOF) {
      CHECK_AVERROR(
          av_interleaved_write_frame(fmt_ctx, nullptr),
          "Failed to flush the encoder.");
      break;
    }
    if (ret == AVERROR(EAGAIN)) {
      break;
    }
    CHECK_AVERROR_NUM(ret, "Failed to fetch encooded packet.");
    // https://github.com/pytorch/audio/issues/2790
    // If this is not set, the last frame is not properly saved, as
    // the encoder cannot figure out when the packet should finish.
    if (packet->duration == 0 && cdc_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
      // 1 means that 1 frame (in codec time base, which is the frame rate)
      // This has to be set before av_packet_rescale_ts bellow.
      packet->duration = 1;
    }
    av_packet_rescale_ts(pkt, cdc_ctx->time_base, stream->time_base);
    packet->stream_index = stream->index;

    ret = av_interleaved_write_frame(fmt_ctx, pkt);
    CHECK_AVERROR_NUM(ret, "Failed to write a packet.");
  } while (ret >= 0);
}

std::pair<Encoder, FilterGraph> get_encode_process(
    const std::string& uri,
    const AVPixelFormat src_fmt,
    int src_width,
    int src_height,
    const EncodeConfig& enc_cfg) {
  const int enc_width = enc_cfg.width > 0 ? enc_cfg.width : src_width;
  const int enc_height = enc_cfg.height > 0 ? enc_cfg.height : src_height;

  auto format_ctx = detail::get_output_format_ctx(uri, enc_cfg.muxer);
  if (format_ctx->oformat->video_codec == AV_CODEC_ID_NONE) {
    SPDL_FAIL(fmt::format(
        "The output format ({}) does not support video/image data.",
        format_ctx->oformat->name));
  }
  const AVCodec* codec =
      get_image_codec(enc_cfg.encoder, format_ctx->oformat, uri);
  const AVPixelFormat enc_fmt = get_enc_fmt(src_fmt, enc_cfg.format, codec);

  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);

  configure_image_codec_ctx(codec_ctx, enc_fmt, enc_width, enc_height, enc_cfg);
  open_codec<MediaType::Image>(codec_ctx.get(), enc_cfg.encoder_options);

  AVStream* stream = create_stream(format_ctx.get(), codec_ctx.get());
  open_format(format_ctx.get(), enc_cfg.muxer_options);

  Encoder encoder{std::move(format_ctx), stream, std::move(codec_ctx)};

  auto filter_graph = get_image_enc_filter(
      src_width,
      src_height,
      src_fmt,
      enc_width,
      enc_height,
      enc_cfg.scale_algo,
      enc_fmt,
      enc_cfg.filter_desc);

  return {std::move(encoder), std::move(filter_graph)};
}
} // namespace spdl::core::detail
