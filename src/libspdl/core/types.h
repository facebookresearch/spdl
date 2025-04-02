/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#define SPDL_DEFAULT_BUFFER_SIZE 8096

#if __has_include(<libavutil/rational.h>)
#include <libavutil/rational.h>
#else
extern "C" {
// Copying the definition of AVRAtional.
// It's unlikely they change, but if it should ever happen, it could cause SEGV.
// https://www.ffmpeg.org/doxygen/4.4/rational_8h_source.html#l00058
// https://www.ffmpeg.org/doxygen/5.1/rational_8h_source.html#l00058
// https://www.ffmpeg.org/doxygen/6.1/rational_8h_source.html#l00058
// https://www.ffmpeg.org/doxygen/7.0/rational_8h_source.html#l00058
typedef struct AVRational {
  int num; ///< Numerator
  int den; ///< Denominator
} AVRational;
}
#endif

namespace spdl::core {

using OptionDict = std::map<std::string, std::string>;

using Rational = AVRational;

// simplified version of AVMediaType so that public headers do not
// include ffmpeg headers
enum class MediaType { Audio, Video, Image };

struct DemuxConfig {
  std::optional<std::string> format = std::nullopt;
  std::optional<OptionDict> format_options = std::nullopt;
  int buffer_size = SPDL_DEFAULT_BUFFER_SIZE;
};

struct DecodeConfig {
  std::optional<std::string> decoder = std::nullopt;
  std::optional<OptionDict> decoder_options = std::nullopt;
};

// Used to construct Dtype when converting buffer to array
enum class ElemClass { Int, UInt, Float };

// Subset of AVCodecID used by nvdec so that we can build NVDEC without
// including FFmpeg headers
enum class CodecID {
  MPEG1VIDEO,
  MPEG2VIDEO,
  MPEG4,
  WMV3,
  VC1,
  H264,
  HEVC,
  VP8,
  VP9,
  MJPEG,
  AV1
};

struct EncodeConfig {
  // Muxer format or device name
  std::optional<std::string> muxer = std::nullopt;
  // Options for muxer
  std::optional<OptionDict> muxer_options = std::nullopt;

  // Encoder
  std::optional<std::string> encoder = std::nullopt;
  // Encoder options
  std::optional<OptionDict> encoder_options = std::nullopt;

  // Pixel/sample format used for encoding
  std::optional<std::string> format = std::nullopt;

  // Rescale options
  int width = -1;
  int height = -1;
  std::optional<std::string> scale_algo = std::nullopt;

  // Optional filter desc
  std::optional<std::string> filter_desc = std::nullopt;

  int bit_rate = -1;
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  int qscale = -1;

  // video
  int gop_size = -1;
  int max_b_frames = -1;
};

template <MediaType media_type>
struct EncodeConfigBase;

template <>
struct EncodeConfigBase<MediaType::Video> {
  int height;
  int width;

  ////////////////////////////////////////////////////////////////////////////////
  // Overrides
  ////////////////////////////////////////////////////////////////////////////////
  const std::optional<std::string> pix_fmt = std::nullopt;
  const std::optional<Rational> frame_rate = std::nullopt;

  int bit_rate = -1;
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  int qscale = -1;

  // video
  int gop_size = -1;
  int max_b_frames = -1;
};

using VideoEncodeConfig = EncodeConfigBase<MediaType::Video>;

// Thrown when unexpected internal error occurs.
class InternalError : public std::logic_error {
  using std::logic_error::logic_error;
};

} // namespace spdl::core
