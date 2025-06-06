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
// Struct to pass around codec info.
// 1. Abstract away AVCodecParameters so that SPDL public header won't require
//    FFmpeg headers
// 2. AVCodecParameters is required when creating decoders, but that's
//    not the case for encoders. Therefore, this class is intended to carry
//    information about codec that should be used for decoding.
template <MediaType media>
class Codec {
  AVCodecParameters* codecpar;

  Rational time_base;
  Rational frame_rate;

 public:
  Codec() = default;
  // From decoding/demuxing context
  Codec(
      const AVCodecParameters*,
      Rational time_base,
      Rational frame_rate) noexcept;
  ~Codec();
  explicit Codec(const Codec<media>&);
  Codec(Codec<media>&&) noexcept;
  Codec<media>& operator=(const Codec<media>&);
  Codec<media>& operator=(Codec<media>&&) noexcept;

  // Getters - common
  std::string get_name() const;
  CodecID get_codec_id() const;

  int64_t get_bit_rate() const;
  int get_bits_per_sample() const;

  Rational get_time_base() const;
  Rational get_frame_rate() const;

  // Note: the returned pointer must not outlive the Codec object
  const AVCodecParameters* get_parameters() const;

  // Getters - Audio specific
  int get_sample_rate() const
    requires(media == MediaType::Audio);
  int get_num_channels() const
    requires(media == MediaType::Audio);
  std::string get_sample_fmt() const
    requires(media == MediaType::Audio);
  std::string get_channel_layout() const
    requires(media == MediaType::Audio);

  // Getters - Video specific
  int get_width() const
    requires(media == MediaType::Video || media == MediaType::Image);
  int get_height() const
    requires(media == MediaType::Video || media == MediaType::Image);
  std::string get_pix_fmt() const
    requires(media == MediaType::Video || media == MediaType::Image);
  Rational get_sample_aspect_ratio() const
    requires(media == MediaType::Video || media == MediaType::Image);
};

using AudioCodec = Codec<MediaType::Audio>;
using VideoCodec = Codec<MediaType::Video>;
using ImageCodec = Codec<MediaType::Image>;

} // namespace spdl::core
