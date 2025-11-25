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

namespace spdl::core::detail {

template <MediaType media>
class EncoderImpl {
  AVCodecContextPtr codec_ctx_;
  int stream_index_;

 public:
  EncoderImpl(AVCodecContextPtr codec_ctx, int stream_index);

  PacketsPtr<media> encode(const FramesPtr<media>&&);
  PacketsPtr<media> flush();

  AVRational get_time_base() const;
  AVCodecParameters* get_codec_par(AVCodecParameters* out = nullptr) const;
  int get_frame_size() const
    requires(media == MediaType::Audio);
};

template <MediaType media>
using EncoderImplPtr = std::unique_ptr<EncoderImpl<media>>;

using AudioEncoderImpl = EncoderImpl<MediaType::Audio>;
using AudioEncoderImplPtr = std::unique_ptr<AudioEncoderImpl>;

using VideoEncoderImpl = EncoderImpl<MediaType::Video>;
using VideoEncoderImplPtr = std::unique_ptr<VideoEncoderImpl>;

template <MediaType media>
EncoderImplPtr<media> make_encoder(
    const AVCodec* codec,
    const EncodeConfigBase<media>& encode_config,
    const std::optional<OptionDict>& encoder_config,
    int stream_index,
    bool global_header = false);

} // namespace spdl::core::detail
