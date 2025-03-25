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
// For now, we storeonly the necessary ones.
// TOD: Generalize it by storing AVCodecParameter
template <MediaType media_type>
class Codec {
  AVCodecParameters* codecpar;

 public:
  explicit Codec(AVCodecParameters*) noexcept;

  std::string get_name() const;
  int get_width() const;
  int get_height() const;
  CodecID get_codec_id() const;
};

using AudioCodec = Codec<MediaType::Audio>;
using VideoCodec = Codec<MediaType::Video>;
using ImageCodec = Codec<MediaType::Image>;

} // namespace spdl::core
