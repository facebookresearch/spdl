/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <string>

struct AVCodecParameters;

namespace spdl::core {
/// Codec information wrapper.
///
/// This class encapsulates codec parameters and provides a media-type-aware
/// interface for accessing codec properties. It abstracts away
/// AVCodecParameters so that SPDL public headers do not require FFmpeg headers.
///
/// This class is primarily intended to carry information about codecs for
/// decoding operations, as AVCodecParameters is required when creating
/// decoders.
template <MediaType media>
class Codec {
  AVCodecParameters* codecpar_;

  Rational time_base_;
  Rational frame_rate_;

 public:
  /// Default constructor.
  Codec() = default;

  /// Construct codec from decoding/demuxing context.
  ///
  /// @param codecpar Codec parameters from FFmpeg.
  /// @param time_base Time base of the stream.
  /// @param frame_rate Frame rate of the stream.
  Codec(
      const AVCodecParameters* codecpar,
      Rational time_base,
      Rational frame_rate) noexcept;

  /// Destructor.
  ~Codec();

  /// Copy constructor.
  explicit Codec(const Codec<media>&);

  /// Move constructor.
  Codec(Codec<media>&&) noexcept;

  /// Copy assignment operator.
  Codec<media>& operator=(const Codec<media>&);

  /// Move assignment operator.
  Codec<media>& operator=(Codec<media>&&) noexcept;

  ////////////////////////////////////////////////////////////////////////////////
  // Common getters
  ////////////////////////////////////////////////////////////////////////////////

  /// Get the codec name.
  /// @return Codec name as a string.
  std::string get_name() const;

  /// Get the codec ID.
  /// @return Codec identifier.
  CodecID get_codec_id() const;

  /// Get the bit rate.
  /// @return Bit rate in bits per second.
  int64_t get_bit_rate() const;

  /// Get bits per sample.
  /// @return Number of bits per sample.
  int get_bits_per_sample() const;

  /// Get the time base.
  /// @return Time base as a rational number.
  Rational get_time_base() const;

  /// Get the frame rate.
  /// @return Frame rate as a rational number.
  Rational get_frame_rate() const;

  /// Get the underlying codec parameters.
  /// @return Pointer to AVCodecParameters. The returned pointer must not
  ///         outlive the Codec object.
  const AVCodecParameters* get_parameters() const;

  ////////////////////////////////////////////////////////////////////////////////
  // Audio-specific getters
  ////////////////////////////////////////////////////////////////////////////////

  /// Get the audio sample rate.
  /// @return Sample rate in Hz.
  int get_sample_rate() const
    requires(media == MediaType::Audio);

  /// Get the number of audio channels.
  /// @return Number of channels.
  int get_num_channels() const
    requires(media == MediaType::Audio);

  /// Get the audio sample format.
  /// @return Sample format as a string.
  std::string get_sample_fmt() const
    requires(media == MediaType::Audio);

  /// Get the audio channel layout.
  /// @return Channel layout as a string.
  std::string get_channel_layout() const
    requires(media == MediaType::Audio);

  ////////////////////////////////////////////////////////////////////////////////
  // Video/Image-specific getters
  ////////////////////////////////////////////////////////////////////////////////

  /// Get the video/image width.
  /// @return Width in pixels.
  int get_width() const
    requires(media == MediaType::Video || media == MediaType::Image);

  /// Get the video/image height.
  /// @return Height in pixels.
  int get_height() const
    requires(media == MediaType::Video || media == MediaType::Image);

  /// Get the pixel format.
  /// @return Pixel format as a string.
  std::string get_pix_fmt() const
    requires(media == MediaType::Video || media == MediaType::Image);

  /// Get the sample aspect ratio.
  /// @return Sample aspect ratio as a rational number.
  Rational get_sample_aspect_ratio() const
    requires(media == MediaType::Video || media == MediaType::Image);
};

/// Audio codec type alias.
using AudioCodec = Codec<MediaType::Audio>;
/// Video codec type alias.
using VideoCodec = Codec<MediaType::Video>;
/// Image codec type alias.
using ImageCodec = Codec<MediaType::Image>;

} // namespace spdl::core
