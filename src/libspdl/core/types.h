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

extern "C" {
#if __has_include(<libavutil/rational.h>)
#include <libavutil/rational.h>
#else
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
#endif
}

namespace spdl::core {

/// Dictionary of options passed to demuxers, decoders, and encoders.
/// Maps option names to their string values.
using OptionDict = std::map<std::string, std::string>;

/// Rational number representation (numerator/denominator).
/// Used for time bases and frame rates.
using Rational = AVRational;

/// Time window specified as a tuple of two Rational values (start, end).
/// Used to specify time ranges for demuxing operations.
using TimeWindow = std::tuple<Rational, Rational>;

/// Media type enumeration.
/// Simplified version of AVMediaType so that public headers do not
/// include FFmpeg headers.
enum class MediaType {
  Audio, ///< Audio media type
  Video, ///< Video media type
  Image ///< Image media type
};

/// Configuration for demuxing operations.
struct DemuxConfig {
  /// Optional format name to force a specific demuxer.
  std::optional<std::string> format = std::nullopt;
  /// Optional format-specific options.
  std::optional<OptionDict> format_options = std::nullopt;
  /// Buffer size for I/O operations.
  int buffer_size = SPDL_DEFAULT_BUFFER_SIZE;
};

/// Configuration for decoding operations.
struct DecodeConfig {
  /// Optional decoder name to force a specific decoder.
  std::optional<std::string> decoder = std::nullopt;
  /// Optional decoder-specific options.
  std::optional<OptionDict> decoder_options = std::nullopt;
};

/// Element class for buffer data types.
/// Used to construct data type information when converting buffer to array.
enum class ElemClass {
  Int, ///< Signed integer
  UInt, ///< Unsigned integer
  Float ///< Floating point
};

/// Codec identifier enumeration.
/// Subset of AVCodecID used by NVDEC so that we can build NVDEC without
/// including FFmpeg headers.
enum class CodecID {
  MPEG1VIDEO, ///< MPEG-1 video codec
  MPEG2VIDEO, ///< MPEG-2 video codec
  MPEG4, ///< MPEG-4 part 2 video codec
  WMV3, ///< Windows Media Video 9 codec
  VC1, ///< VC-1 video codec
  H264, ///< H.264/AVC video codec
  HEVC, ///< H.265/HEVC video codec
  VP8, ///< VP8 video codec
  VP9, ///< VP9 video codec
  MJPEG, ///< Motion JPEG codec
  AV1 ///< AV1 video codec
};

/// Base configuration template for encoding operations.
/// Specialized for Video and Audio media types.
template <MediaType media>
struct EncodeConfigBase;

/// Video encoding configuration.
template <>
struct EncodeConfigBase<MediaType::Video> {
  /// Output video height in pixels.
  int height;
  /// Output video width in pixels.
  int width;

  ////////////////////////////////////////////////////////////////////////////////
  // Optional overrides
  ////////////////////////////////////////////////////////////////////////////////
  /// Pixel format override. If not specified, uses encoder default.
  const std::optional<std::string> pix_fmt = std::nullopt;
  /// Frame rate override. If not specified, uses encoder default.
  const std::optional<Rational> frame_rate = std::nullopt;

  /// Target bit rate in bits per second. Negative values use encoder default.
  int bit_rate = -1;
  /// Compression level. Negative values use encoder default.
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  int qscale = -1;

  /// Group of Pictures (GOP) size. Negative values use encoder default.
  int gop_size = -1;
  /// Maximum number of B-frames. Negative values use encoder default.
  int max_b_frames = -1;

  /// Optional colorspace specification.
  std::optional<std::string> colorspace;
  /// Optional color primaries specification.
  std::optional<std::string> color_primaries;
  /// Optional color transfer characteristics specification.
  std::optional<std::string> color_trc;
};

/// Audio encoding configuration.
template <>
struct EncodeConfigBase<MediaType::Audio> {
  /// Number of audio channels.
  int num_channels;

  ////////////////////////////////////////////////////////////////////////////////
  // Optional overrides
  ////////////////////////////////////////////////////////////////////////////////
  /// Sample format override. If not specified, uses encoder default.
  const std::optional<std::string> sample_fmt = std::nullopt;
  /// Sample rate override. If not specified, uses encoder default.
  const std::optional<int> sample_rate = std::nullopt;

  /// Target bit rate in bits per second. Negative values use encoder default.
  int bit_rate = -1;
  /// Compression level. Negative values use encoder default.
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  int qscale = -1;
};

/// Video encoding configuration type alias.
using VideoEncodeConfig = EncodeConfigBase<MediaType::Video>;
/// Audio encoding configuration type alias.
using AudioEncodeConfig = EncodeConfigBase<MediaType::Audio>;

/// Exception thrown when unexpected internal error occurs.
class InternalError : public std::logic_error {
  using std::logic_error::logic_error;
};

} // namespace spdl::core
