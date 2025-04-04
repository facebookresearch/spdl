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
          "Codec `{}` does not support the frame rate `{}/{}`. "
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
      "Codec `{}` does not have a default frame rate. Please specify one.",
      codec->name));
}

AVCodecContextPtr get_codec_context(
    const AVCodec* codec,
    const VideoEncodeConfig& encode_config) {
  // Check before allocating AVCodecContext
  auto pix_fmt = get_pix_fmt(codec, encode_config.pix_fmt);
  auto frame_rate = get_frame_rate(codec, encode_config.frame_rate);

  auto ctx = AVCodecContextPtr{CHECK_AVALLOCATE(avcodec_alloc_context3(codec))};

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
  if (encode_config.colorspace) {
    auto& name = encode_config.colorspace.value();
    int val = av_color_space_from_name(name.c_str());
    CHECK_AVERROR_NUM(val, fmt::format("Unexpected color space: {}", name));
    ctx->colorspace = (AVColorSpace)val;
  }
  if (encode_config.color_primaries) {
    auto& name = encode_config.color_primaries.value();
    int val = av_color_primaries_from_name(name.c_str());
    CHECK_AVERROR_NUM(val, fmt::format("Unexpected color primaries: {}", name));
    ctx->color_primaries = (AVColorPrimaries)val;
  }
  if (encode_config.color_trc) {
    auto& name = encode_config.color_trc.value();
    int val = av_color_transfer_from_name(name.c_str());
    CHECK_AVERROR_NUM(
        val, fmt::format("Unexpected color transfer characteristic: {}", name));
    ctx->color_trc = (AVColorTransferCharacteristic)val;
  }

  return ctx;
}

bool is_sample_fmt_supported(AVSampleFormat fmt, const AVSampleFormat* fmts) {
  if (!fmts) {
    return true;
  }
  while (*fmts != AV_SAMPLE_FMT_NONE) {
    if (fmt == *fmts) {
      return true;
    }
    ++fmts;
  }
  return false;
}

std::vector<std::string> to_str(const AVSampleFormat* fmts) {
  std::vector<std::string> ret;
  for (; *fmts != AV_SAMPLE_FMT_NONE; ++fmts) {
    ret.emplace_back(av_get_sample_fmt_name(*fmts));
  }
  return ret;
}

AVSampleFormat get_sample_fmt(
    const AVCodec* codec,
    const std::optional<std::string>& override) {
  if (override) {
    auto fmt = av_get_sample_fmt(override.value().c_str());
    if (fmt == AV_SAMPLE_FMT_NONE) {
      SPDL_FAIL(fmt::format("Unexpected sample format: ", override.value()));
    }
    if (!is_sample_fmt_supported(fmt, codec->sample_fmts)) {
      SPDL_FAIL(fmt::format(
          "Codec `{}` does not support the sample format `{}`. Supported values are {}",
          codec->name,
          override.value(),
          fmt::join(to_str(codec->sample_fmts), ", ")));
    }
    return fmt;
  }
  if (codec->sample_fmts) {
    return codec->sample_fmts[0];
  }
  SPDL_FAIL(fmt::format(
      "Codec `{}` does not have a default sample format. Please specify one.",
      codec->name));
}

bool is_sample_rate_supported(int rate, const int* rates) {
  if (!rates) {
    return true;
  }
  for (; *rates; ++rates) {
    if (rate == *rates) {
      return true;
    }
  }
  return false;
}

std::vector<int> to_str(const int* rates) {
  std::vector<int> ret;
  for (; *rates != AV_SAMPLE_FMT_NONE; ++rates) {
    ret.push_back(*rates);
  }
  return ret;
}

int get_sample_rate(const AVCodec* codec, const std::optional<int>& override) {
  // G.722 only supports 16000 Hz, but it does not list the sample rate in
  // supported_samplerates so we hard code it here.
  if (codec->id == AV_CODEC_ID_ADPCM_G722) {
    if (override && *override != 16'000) {
      SPDL_FAIL(fmt::format(
          "Codec `{}` does not support the sample rate `{}`. "
          "Supported values are 16000.",
          codec->name,
          *override));
    }
    return 16'000;
  }

  if (override) {
    auto rate = override.value();
    if (rate <= 0) {
      SPDL_FAIL(
          fmt::format("Sample rate must be greater than 0. Found: {}", rate));
    }
    if (!is_sample_rate_supported(rate, codec->supported_samplerates)) {
      SPDL_FAIL(fmt::format(
          "Codec `{}` does not support the sample rate `{}`. Supported values are {}",
          codec->name,
          rate,
          fmt::join(to_str(codec->supported_samplerates), ", ")));
    }
    return rate;
  }
  if (codec->supported_samplerates) {
    return codec->supported_samplerates[0];
  }
  SPDL_FAIL(fmt::format(
      "Codec `{}` does not have a default sample rate. Please specify one.",
      codec->name));
}

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 17, 100)
std::vector<std::string> to_str(const AVChannelLayout* p) {
  std::vector<std::string> ret;
#define BUF_SIZE 64
  char buf[BUF_SIZE];
  while (p->nb_channels) {
    auto size = av_channel_layout_describe(p, buf, BUF_SIZE);
    CHECK_AVERROR_NUM(size, "Failed to fetch a channel layout name.");
    if (size > BUF_SIZE) {
      size = BUF_SIZE;
    }
    ret.emplace_back(std::string{buf, static_cast<size_t>(size)});
    p++;
  }
#undef BUF_SIZE
  return ret;
}
void set_channels(AVCodecContext* ctx, int num_channels) {
  const auto* codec = ctx->codec;
  if (!codec->ch_layouts) {
    av_channel_layout_default(&ctx->ch_layout, num_channels);
    return;
  }
  const auto* p = codec->ch_layouts;
  while (p->nb_channels) {
    if (p->nb_channels == num_channels) {
      CHECK_AVERROR(
          av_channel_layout_copy(&ctx->ch_layout, p),
          "Failed to copy channel layout.");
      return;
    }
    p++;
  }
  SPDL_FAIL(fmt::format(
      "Codec `{}` does not support {} channels. Supported values are {}.",
      codec->name,
      num_channels,
      fmt::join(to_str(codec->ch_layouts), ", ")));
}
#else
std::vector<std::string> to_str(const uint64_t* layouts) {
  std::vector<std::string> ret;
  for (; *layouts; ++layouts) {
    ret.emplace_back(av_get_channel_name(*layouts));
  }
  return ret;
}

int64_t get_channel_layout(const AVCodec* codec, int num_channels) {
  if (!codec->channel_layouts) {
    return av_get_default_channel_layout(num_channels);
  }
  for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
    if (av_get_channel_layout_nb_channels(*it) == num_channels) {
      return *it;
    }
  }
  SPDL_FAIL(fmt::format(
      "Codec `{}` does not support {} channels. Supported values are {}",
      codec->name,
      num_channels,
      fmt::join(to_str(codec->channel_layouts), ", ")));
}

void set_channels(AVCodecContext* ctx, int num_channels) {
  ctx->channel_layout = get_channel_layout(ctx->codec, num_channels);
  ctx->channels = num_channels;
}
#endif

AVCodecContextPtr get_codec_context(
    const AVCodec* codec,
    const AudioEncodeConfig& encode_config) {
  // Check before allocating AVCodecContext
  auto sample_fmt = get_sample_fmt(codec, encode_config.sample_fmt);
  auto sample_rate = get_sample_rate(codec, encode_config.sample_rate);

  auto ctx = AVCodecContextPtr{CHECK_AVALLOCATE(avcodec_alloc_context3(codec))};

  ctx->sample_fmt = sample_fmt;
  ctx->sample_rate = sample_rate;
  ctx->time_base = AVRational{1, sample_rate};
  set_channels(ctx.get(), encode_config.num_channels);

  // Set optional stuff
  if (encode_config.bit_rate > 0) {
    ctx->bit_rate = encode_config.bit_rate;
  }
  if (encode_config.compression_level != -1) {
    ctx->compression_level = encode_config.compression_level;
  }
  if (encode_config.qscale >= 0) {
    ctx->flags |= AV_CODEC_FLAG_QSCALE;
    ctx->global_quality = FF_QP2LAMBDA * encode_config.qscale;
  }
  return ctx;
}
} // namespace
template <MediaType media>
EncoderImpl<media>::EncoderImpl(AVCodecContextPtr codec_ctx_, int index)
    : codec_ctx(std::move(codec_ctx_)), stream_index(index) {}

template <MediaType media>
AVRational EncoderImpl<media>::get_time_base() const {
  return codec_ctx->time_base;
}

template <MediaType media>
AVCodecParameters* EncoderImpl<media>::get_codec_par(
    AVCodecParameters* out) const {
  if (!out) {
    out = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  }

  CHECK_AVERROR(
      avcodec_parameters_from_context(out, codec_ctx.get()),
      "Failed to copy codec context.");
  return out;
}

template <MediaType media>
EncoderImplPtr<media> make_encoder(
    const AVCodec* codec,
    const EncodeConfigBase<media>& encode_config,
    const std::optional<OptionDict>& encoder_config,
    int stream_index,
    bool global_header) {
  auto codec_ctx = get_codec_context(codec, encode_config);
  if (global_header) {
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  open_codec_for_encode(codec_ctx.get(), encoder_config);
  return std::make_unique<EncoderImpl<media>>(
      std::move(codec_ctx), stream_index);
}

template AudioEncoderImplPtr make_encoder(
    const AVCodec* codec,
    const AudioEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config,
    int stream_index,
    bool global_header);

template VideoEncoderImplPtr make_encoder(
    const AVCodec* codec,
    const VideoEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config,
    int stream_index,
    bool global_header);

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

template <MediaType media>
PacketsPtr<media> EncoderImpl<media>::encode(const FramesPtr<media>&& frames) {
  auto ret = std::make_unique<Packets<media>>(
      frames->get_id(), stream_index, codec_ctx->time_base);
  auto encoding = _encode(codec_ctx.get(), frames->get_frames(), false);
  while (encoding) {
    ret->pkts.push(encoding().release());
  }
  return ret;
}

template <MediaType media>
PacketsPtr<media> EncoderImpl<media>::flush() {
  auto ret =
      std::make_unique<Packets<media>>(0, stream_index, codec_ctx->time_base);
  std::vector<AVFrame*> dummy{};
  auto encoding = _encode(codec_ctx.get(), dummy, true);
  while (encoding) {
    ret->pkts.push(encoding().release());
  }
  return ret;
}

template class EncoderImpl<MediaType::Audio>;
template class EncoderImpl<MediaType::Video>;
}; // namespace spdl::core::detail
