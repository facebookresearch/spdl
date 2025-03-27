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
//
// Note: Currently Demuxer and Packets individually creates Codec class
// as requested by client code.
// Packets manages its own AVCodecParameters lifecycle.
// Perhaps it might be good to unify and let Packets hold Codec class?
template <MediaType media_type>
class Codec {
  AVCodecParameters* codecpar;

 public:
  Rational time_base;
  Rational frame_rate;

 public:
  Codec(const AVCodecParameters*, Rational, Rational = {1, 1}) noexcept;
  ~Codec();
  Codec(const Codec<media_type>&);
  Codec(Codec<media_type>&&) noexcept;
  Codec<media_type>& operator=(const Codec<media_type>&);
  Codec<media_type>& operator=(Codec<media_type>&&) noexcept;

  std::string get_name() const;
  int get_sample_rate() const
    requires(media_type == MediaType::Audio);
  int get_num_channels() const
    requires(media_type == MediaType::Audio);
  int get_width() const
    requires(media_type == MediaType::Video || media_type == MediaType::Image);
  int get_height() const
    requires(media_type == MediaType::Video || media_type == MediaType::Image);
  CodecID get_codec_id() const;

  // Note: the returned pointer must not outlive the Codec object
  const AVCodecParameters* get_parameters() const;
};

using AudioCodec = Codec<MediaType::Audio>;
using VideoCodec = Codec<MediaType::Video>;
using ImageCodec = Codec<MediaType::Image>;

} // namespace spdl::core
