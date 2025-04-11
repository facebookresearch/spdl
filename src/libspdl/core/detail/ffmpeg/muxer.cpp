/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/muxer.h"

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>
#include <fmt/format.h>

#include <filesystem>
#include <set>

namespace spdl::core::detail {
namespace {} // namespace

MuxerImpl::MuxerImpl(
    const std::string& uri,
    const std::optional<std::string>& format)
    : fmt_ctx(get_output_format_ctx(uri, format)) {}

namespace {
template <MediaType media>
const AVCodec* get_codec(
    const AVFormatContext* fmt_ctx,
    const std::optional<std::string>& override) {
  if (override) {
    auto name = override.value();
    const AVCodec* c = avcodec_find_encoder_by_name(name.c_str());
    if (!c) [[unlikely]] {
      SPDL_FAIL(fmt::format("Unknown codec: {}", override.value()));
    }
    if constexpr (media == MediaType::Audio) {
      if (c->type != AVMEDIA_TYPE_AUDIO) [[unlikely]] {
        SPDL_FAIL(fmt::format("Codec `{}` is not audio type", name));
      }
    }
    if constexpr (media == MediaType::Video) {
      if (c->type != AVMEDIA_TYPE_VIDEO) [[unlikely]] {
        SPDL_FAIL(fmt::format("Codec `{}` is not video type", name));
      }
    }
    return c;
  }

  // Special handling for Image
  // FFmpeg defaults to 'image2' muxer, of which default encoder is MJPEG.
  // This also applies to formats like PNG and TIFF
  if (std::strcmp(fmt_ctx->oformat->name, "image2") == 0) {
    // The following list was obtained by running
    // ffmpeg -h muxer=image2 | grep 'Common extensions'
    // then for each extension, checking the encoder by
    // fmpeg -hide_banner -h encoder=$ext | grep 'Encoder '
    static const std::set<std::string> exts{
        "bmp", "dpx",    "exr", "pam",   "pbm", "pcx", "pfm",
        "pgm", "pgmyuv", "phm", "png",   "ppm", "sgi", "tiff",
        "xwd", "vbn",    "xbm", "xface", "qoi", "hdr", "wbmp"};

    auto ext = std::filesystem::path(fmt_ctx->url).extension().string();
    if (!ext.empty()) {
      ext = ext.substr(1);
      if (exts.contains(ext)) {
        const AVCodec* c = avcodec_find_encoder_by_name(ext.c_str());
        if (c) {
          return c;
        }
      }
    }
  }

  AVCodecID default_codec;
  if constexpr (media == MediaType::Video || media == MediaType::Image) {
    default_codec = fmt_ctx->oformat->video_codec;
  }
  if constexpr (media == MediaType::Audio) {
    default_codec = fmt_ctx->oformat->audio_codec;
  }
  const AVCodec* c = avcodec_find_encoder(default_codec);
  if (!c) [[unlikely]] {
    SPDL_FAIL(fmt::format(
        "The `{}` codec does not have a default encoder. Please specify one.",
        avcodec_get_name(default_codec)));
  }
  return c;
}
} // namespace

template <MediaType media>
void MuxerImpl::assert_media_is_supported() const {
  if constexpr (media == MediaType::Audio) {
    if (fmt_ctx->oformat->audio_codec == AV_CODEC_ID_NONE) [[unlikely]] {
      SPDL_FAIL(
          fmt::format("`{}` does not support audio.", fmt_ctx->oformat->name));
    }
  }
  if constexpr (media == MediaType::Video) {
    if (fmt_ctx->oformat->video_codec == AV_CODEC_ID_NONE) [[unlikely]] {
      SPDL_FAIL(
          fmt::format("`{}` does not support video.", fmt_ctx->oformat->name));
    }
  }
}

template <MediaType media>
EncoderImplPtr<media> MuxerImpl::add_encode_stream(
    const EncodeConfigBase<media>& codec_config,
    const std::optional<std::string>& encoder_name,
    const std::optional<OptionDict>& encoder_config) {
  assert_media_is_supported<media>();

  const AVCodec* codec = get_codec<media>(fmt_ctx.get(), encoder_name);
  auto ret = make_encoder(
      codec,
      codec_config,
      encoder_config,
      fmt_ctx->nb_streams,
      fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER);

  // Register the stream
  AVStream* s = CHECK_AVALLOCATE(avformat_new_stream(fmt_ctx.get(), nullptr));

  // Note: time base may be adjusted when `calling avformat_write_header()`
  // https://ffmpeg.org/doxygen/5.1/structAVStream.html#a9db755451f14e2bf590d4b85d82b32e6
  s->time_base = ret->get_time_base();
  ret->get_codec_par(s->codecpar);

  return ret;
}

template AudioEncoderImplPtr MuxerImpl::add_encode_stream(
    const AudioEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);

template VideoEncoderImplPtr MuxerImpl::add_encode_stream(
    const VideoEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);

template <MediaType media>
void MuxerImpl::add_remux_stream(const Codec<media>& codec) {
  assert_media_is_supported<media>();

  AVStream* s = CHECK_AVALLOCATE(avformat_new_stream(fmt_ctx.get(), nullptr));
  s->time_base = codec.get_time_base();
  CHECK_AVERROR(
      avcodec_parameters_copy(s->codecpar, codec.get_parameters()),
      "Failed to copy codec context.");
}

template void MuxerImpl::add_remux_stream(const AudioCodec& codec);
template void MuxerImpl::add_remux_stream(const VideoCodec& codec);

void MuxerImpl::open(const std::optional<OptionDict>& muxer_config) {
  open_format(fmt_ctx.get(), muxer_config);
}

void MuxerImpl::write(
    int i,
    const std::vector<AVPacket*>& packets,
    AVRational time_base) {
  if (i < 0 || fmt_ctx->nb_streams <= i) {
    SPDL_FAIL(fmt::format(
        "The stream index ({}) is out of bound. (0, {}]",
        i,
        fmt_ctx->nb_streams));
  }
  AVStream* s = fmt_ctx->streams[i];
  for (auto* p : packets) {
    av_packet_rescale_ts(p, time_base, s->time_base);
    p->stream_index = s->index;
    CHECK_AVERROR(
        av_interleaved_write_frame(fmt_ctx.get(), p),
        "Failed to write a packet.");
  }
}

void MuxerImpl::flush() {
  CHECK_AVERROR(
      av_interleaved_write_frame(fmt_ctx.get(), nullptr), "Failed to flush.");
}

void MuxerImpl::close() {
  CHECK_AVERROR(av_write_trailer(fmt_ctx.get()), "Failed to write trailer.");
  // Close the file if it was not provided by client code (i.e. when not
  // file-like object)
  AVFORMAT_CONST AVOutputFormat* fmt = fmt_ctx->oformat;
  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(fmt_ctx->flags & AVFMT_FLAG_CUSTOM_IO)) {
    // avio_closep can be only applied to AVIOContext opened by avio_open
    avio_closep(&(fmt_ctx->pb));
  }
}

} // namespace spdl::core::detail
