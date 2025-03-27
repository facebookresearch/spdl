/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/codec.h>

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"

#include "fmt/core.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace spdl::core {
namespace {
inline AVCodecParameters* copy(const AVCodecParameters* src) {
  auto dst = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  CHECK_AVERROR(
      avcodec_parameters_copy(dst, src), "Failed to copy codec parameters.");
  return dst;
}
} // namespace

template <MediaType media_type>
Codec<media_type>::Codec(
    const AVCodecParameters* p,
    Rational time_base,
    Rational frame_rate) noexcept
    : codecpar(copy(p)), time_base(time_base), frame_rate(frame_rate) {}

template <MediaType media_type>
Codec<media_type>::Codec(const Codec<media_type>& other)
    : codecpar(copy(other.codecpar)),
      time_base(other.time_base),
      frame_rate(other.frame_rate) {}

template <MediaType media_type>
Codec<media_type>& Codec<media_type>::operator=(
    const Codec<media_type>& other) {
  codecpar = copy(other.codecpar);
  time_base = other.time_base;
  frame_rate = other.frame_rate;
  return *this;
}

template <MediaType media_type>
Codec<media_type>::Codec(Codec<media_type>&& other) noexcept
    : codecpar(nullptr), time_base({1, 1}), frame_rate({1, 1}) {
  *this = std::move(other);
}

template <MediaType media_type>
Codec<media_type>& Codec<media_type>::operator=(
    Codec<media_type>&& other) noexcept {
  using std::swap;
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  return *this;
}

template <MediaType media_type>
Codec<media_type>::~Codec() {
  avcodec_parameters_free(&codecpar);
}

template <MediaType media_type>
std::string Codec<media_type>::get_name() const {
  return std::string(avcodec_get_name(codecpar->codec_id));
}

template <MediaType media_type>
int Codec<media_type>::get_width() const
  requires(media_type == MediaType::Video || media_type == MediaType::Image)
{
  return codecpar->width;
}

template <MediaType media_type>
int Codec<media_type>::get_height() const
  requires(media_type == MediaType::Video || media_type == MediaType::Image)
{
  return codecpar->height;
}

template <MediaType media_type>
int Codec<media_type>::get_sample_rate() const
  requires(media_type == MediaType::Audio)
{
  return codecpar->sample_rate;
}

template <MediaType media_type>
int Codec<media_type>::get_num_channels() const
  requires(media_type == MediaType::Audio)
{
  return GET_NUM_CHANNELS(codecpar);
}

template <MediaType media_type>
CodecID Codec<media_type>::get_codec_id() const {
  // NOTE: Currently only those used by NVDEC are handled.
  switch (codecpar->codec_id) {
    case AV_CODEC_ID_MPEG1VIDEO:
      return CodecID::MPEG1VIDEO;
    case AV_CODEC_ID_MPEG2VIDEO:
      return CodecID::MPEG2VIDEO;
    case AV_CODEC_ID_MPEG4:
      return CodecID::MPEG4;

    case AV_CODEC_ID_WMV3:
      return CodecID::WMV3;
    case AV_CODEC_ID_VC1:
      return CodecID::VC1;
    case AV_CODEC_ID_H264:
      return CodecID::H264;
    case AV_CODEC_ID_HEVC:
      return CodecID::HEVC;
    case AV_CODEC_ID_VP8:
      return CodecID::VP8;
    case AV_CODEC_ID_VP9:
      return CodecID::VP9;
    case AV_CODEC_ID_MJPEG:
      return CodecID::MJPEG;
    case AV_CODEC_ID_AV1:
      return CodecID::AV1;
    default:
      SPDL_FAIL(fmt::format(
          "Unsupported codec ID: {}", avcodec_get_name(codecpar->codec_id)));
  }
}

template <MediaType media_type>
const AVCodecParameters* Codec<media_type>::get_parameters() const {
  return codecpar;
}

template class Codec<MediaType::Audio>;
template class Codec<MediaType::Video>;
template class Codec<MediaType::Image>;

} // namespace spdl::core
