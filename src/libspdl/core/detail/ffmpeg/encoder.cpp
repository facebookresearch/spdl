/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/encoder.h"

#include <libspdl/core/generator.h>
#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>
#include <fmt/format.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace spdl::core::detail {
namespace {
const AVCodec* get_codec(const std::string& codec_name) {
  auto* p = avcodec_find_encoder_by_name(codec_name.c_str());
  if (!p) {
    SPDL_FAIL(fmt::format("No codec found: `{}`", codec_name));
  }
  return p;
}

bool is_pix_fmt_supported(
    const AVPixelFormat fmt,
    const AVPixelFormat* pix_fmts) {
  if (!pix_fmts) {
    return true;
  }
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (fmt == *pix_fmts) {
      return true;
    }
    ++pix_fmts;
  }
  return false;
}

std::string to_str(const AVPixelFormat* pix_fmts) {
  std::vector<std::string> ret;
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    ret.emplace_back(av_get_pix_fmt_name(*pix_fmts));
    ++pix_fmts;
  }
  return fmt::to_string(fmt::join(ret, ", "));
}

AVPixelFormat get_pix_fmt(
    const AVCodec* codec,
    const std::optional<std::string>& override) {
  if (override) {
    auto fmt = av_get_pix_fmt(override.value().c_str());
    if (!is_pix_fmt_supported(fmt, codec->pix_fmts)) {
      SPDL_FAIL(fmt::format(
          "`{}` does not support the pixel format `{}`. "
          "Supported values are {}",
          codec->name,
          override.value(),
          to_str(codec->pix_fmts)));
    }
    return fmt;
  }
  if (codec->pix_fmts) {
    return codec->pix_fmts[0];
  }
  SPDL_FAIL(fmt::format(
      "`{}` does not have a default pixel format. Please specify one.",
      codec->name));
}

bool is_frame_rate_supported(AVRational rate, const AVRational* rates) {
  if (!rates) {
    return true;
  }
  for (; !(rates->num == 0 && rates->den == 0); ++rates) {
    if (av_cmp_q(rate, *rates) == 0) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> to_str(const AVRational* rates) {
  std::vector<std::string> ret;
  for (; !(rates->num == 0 && rates->den == 0); ++rates) {
    ret.push_back(fmt::format("{}/{}", rates->num, rates->den));
  }
  return ret;
}

AVRational get_frame_rate(
    const AVCodec* codec,
    const std::optional<Rational>& override) {
  if (override) {
    const auto& rate = override.value();
    if (rate.num <= 0 || rate.den <= 0) {
      SPDL_FAIL(fmt::format(
          "Frame rate must be positive finite. Found: {}/{}",
          rate.num,
          rate.den));
    }
    if (!is_frame_rate_supported(rate, codec->supported_framerates)) {
      SPDL_FAIL(fmt::format(
          "`{}` does not support the frame rate `{}/{}`. "
          "Supported values are {}",
          codec->name,
          rate.num,
          rate.den,
          fmt::join(to_str(codec->supported_framerates), ", ")));
    }
    return rate;
  }
  if (codec->supported_framerates) {
    return codec->supported_framerates[0];
  }
  SPDL_FAIL(fmt::format(
      "`{}` does not have a default frame rate. Please specify one.",
      codec->name));
}

AVCodecContextPtr get_codec_context(
    const AVCodec* codec,
    const VideoEncodeConfig& encode_config) {
  // Check before allocating a bare pointer
  auto pix_fmt = get_pix_fmt(codec, encode_config.pix_fmt);
  auto frame_rate = get_frame_rate(codec, encode_config.frame_rate);

  auto ctx = AVCodecContextPtr{CHECK_AVALLOCATE(avcodec_alloc_context3(codec))};
  AVRational av_stream_get_r_frame_rate(const AVStream* s);

  ctx->pix_fmt = pix_fmt;
  ctx->width = encode_config.width;
  ctx->height = encode_config.height;
  ctx->framerate = frame_rate;
  ctx->time_base = av_inv_q(ctx->framerate);

  if (encode_config.bit_rate > 0) {
    ctx->bit_rate = encode_config.bit_rate;
  }
  if (encode_config.compression_level != -1) {
    ctx->compression_level = encode_config.compression_level;
  }
  if (encode_config.gop_size != -1) {
    ctx->gop_size = encode_config.gop_size;
  }
  if (encode_config.max_b_frames != -1) {
    ctx->max_b_frames = encode_config.max_b_frames;
  }
  if (encode_config.qscale >= 0) {
    ctx->flags |= AV_CODEC_FLAG_QSCALE;
    ctx->global_quality = FF_QP2LAMBDA * encode_config.qscale;
  }
  return ctx;
}
} // namespace
template <MediaType media_type>
EncoderImpl<media_type>::EncoderImpl(AVCodecContextPtr codec_ctx_)
    : codec_ctx(std::move(codec_ctx_)) {}

template <MediaType media_type>
AVRational EncoderImpl<media_type>::get_time_base() const {
  return codec_ctx->time_base;
}

template <MediaType media_type>
AVCodecParameters* EncoderImpl<media_type>::get_codec_par(
    AVCodecParameters* out) const {
  if (!out) {
    out = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  }

  CHECK_AVERROR(
      avcodec_parameters_from_context(out, codec_ctx.get()),
      "Failed to copy codec context.");
  return out;
}

std::unique_ptr<VideoEncoderImpl> make_encoder(
    const std::string& encoder_name,
    const VideoEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config) {
  const AVCodec* codec = get_codec(encoder_name);
  return make_encoder(codec, encode_config, encoder_config, false);
}

std::unique_ptr<VideoEncoderImpl> make_encoder(
    const AVCodec* codec,
    const VideoEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config,
    bool global_header) {
  auto codec_ctx = get_codec_context(codec, encode_config);
  if (global_header) {
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  open_codec<MediaType::Video>(codec_ctx.get(), encoder_config);
  return std::make_unique<VideoEncoderImpl>(std::move(codec_ctx));
}

namespace {

Generator<AVFrame*> stream_frame(
    const std::vector<AVFrame*>& frames,
    bool flush) {
  for (auto* f : frames) {
    co_yield f;
  }
  if (flush) {
    co_yield nullptr;
  }
}

Generator<AVPacketPtr> _encode(
    AVCodecContext* codec_ctx,
    const std::vector<AVFrame*>& frames,
    bool flush) {
  int ret = 0;
  auto frame_stream = stream_frame(frames, flush);
  while (frame_stream) {
    ret = avcodec_send_frame(codec_ctx, frame_stream());
    CHECK_AVERROR_NUM(ret, "Failed to send frame to encode context.");
    while (ret >= 0) {
      auto pkt = AVPacketPtr{CHECK_AVALLOCATE(av_packet_alloc())};
      ret = avcodec_receive_packet(codec_ctx, pkt.get());
      switch (ret) {
        case AVERROR_EOF:
          co_return;
        case AVERROR(EAGAIN):
          break;
        default: {
          CHECK_AVERROR_NUM(ret, "Failed to fetch encooded packet.");
          // https://github.com/pytorch/audio/issues/2790
          // If this is not set, the last frame is not properly saved, as
          // the encoder cannot figure out when the packet should finish.
          if (pkt->duration == 0 &&
              codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
            // 1 means that 1 frame (in codec time base, which is the frame
            // rate) This has to be set before av_packet_rescale_ts bellow.
            pkt->duration = 1;
          }
          co_yield std::move(pkt);
        }
      }
    }
  }
}

} // namespace

template <MediaType media_type>
PacketsPtr<media_type> EncoderImpl<media_type>::encode(
    const FFmpegFramesPtr<media_type>&& frames) {
  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      fmt::format("{}", frames->get_id()),
      VideoCodec{get_codec_par(), codec_ctx->time_base, codec_ctx->framerate});
  auto encoding = _encode(codec_ctx.get(), frames->get_frames(), false);
  while (encoding) {
    auto pkt = encoding();
    ret->push(pkt.release());
  }
  return ret;
}

template <MediaType media_type>
PacketsPtr<media_type> EncoderImpl<media_type>::flush() {
  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      "flush",
      VideoCodec{get_codec_par(), codec_ctx->time_base, codec_ctx->framerate});
  std::vector<AVFrame*> dummy{};
  auto encoding = _encode(codec_ctx.get(), dummy, true);
  while (encoding) {
    auto pkt = encoding();
    ret->push(pkt.release());
  }
  return ret;
}

template class EncoderImpl<MediaType::Video>;
}; // namespace spdl::core::detail
