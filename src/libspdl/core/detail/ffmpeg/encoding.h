/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

class Encoder {
  AVFormatOutputContextPtr format_ctx;
  AVStream* stream;
  AVCodecContextPtr codec_ctx;
  AVPacketPtr packet;

 public:
  Encoder(
      AVFormatOutputContextPtr&& format_ctx_,
      AVStream* stream_,
      AVCodecContextPtr&& codec_ctx_);
  void encode(const AVFramePtr& frame);
};

std::pair<Encoder, FilterGraph> get_encode_process(
    const std::string& uri,
    const AVPixelFormat src_fmt,
    int src_width,
    int src_height,
    const EncodeConfig& enc_cfg);

} // namespace spdl::core::detail
