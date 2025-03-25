/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/generator.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

// Wraps AVCodecContextPtr and provide convenient methods
struct DecoderCore {
  AVCodecContextPtr codec_ctx;

  DecoderCore(
      const AVCodecParameters*,
      Rational time_base,
      const std::optional<DecodeConfig>& cfg = std::nullopt);

  Generator<AVFramePtr> decode(AVPacket*, bool flush_null = false);
};

Generator<AVFramePtr> decode_packets(
    const std::vector<AVPacket*>& packets,
    DecoderCore& decoder,
    std::optional<FilterGraph>& filter);

} // namespace spdl::core::detail
