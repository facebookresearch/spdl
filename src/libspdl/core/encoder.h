/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <memory>
#include <string>

namespace spdl::core {

struct VideoEncodeConfig {
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

namespace detail {
template <MediaType media_type>
class EncoderImpl;
}

class VideoEncoder {
  detail::EncoderImpl<MediaType::Video>* pImpl;

 public:
  VideoEncoder(detail::EncoderImpl<MediaType::Video>*);
  VideoEncoder(const VideoEncoder&) = delete;
  VideoEncoder& operator=(const VideoEncoder&) = delete;
  VideoEncoder(VideoEncoder&&) = delete;
  VideoEncoder& operator=(VideoEncoder&&) = delete;

  ~VideoEncoder();

  VideoPacketsPtr encode(const FFmpegVideoFramesPtr&&);

  VideoPacketsPtr flush();
};

using VideoEncoderPtr = std::unique_ptr<VideoEncoder>;

} // namespace spdl::core
