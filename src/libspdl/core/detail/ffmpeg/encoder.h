/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/encoder.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <memory>
#include <string>

namespace spdl::core::detail {

template <MediaType media_type>
class EncoderImpl {
  AVCodecContextPtr codec_ctx;

 public:
  EncoderImpl(AVCodecContextPtr codec_ctx);

  PacketsPtr<media_type> encode(const FFmpegFramesPtr<media_type>&&);
  PacketsPtr<media_type> flush();

  AVRational get_time_base() const;
  AVCodecParameters* get_codec_par(AVCodecParameters* out = nullptr) const;
};

using VideoEncoderImpl = EncoderImpl<MediaType::Video>;

// temp; to be removed.
std::unique_ptr<VideoEncoderImpl> make_encoder(
    const std::string& encoder_name,
    const VideoEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config);

std::unique_ptr<VideoEncoderImpl> make_encoder(
    const AVCodec* codec,
    const VideoEncodeConfig& encode_config,
    const std::optional<OptionDict>& encoder_config,
    bool global_header = false);

} // namespace spdl::core::detail
