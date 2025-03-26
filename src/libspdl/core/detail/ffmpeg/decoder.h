/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {
////////////////////////////////////////////////////////////////////////////////
// DecoderImpl
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
class DecoderImpl {
  AVCodecContextPtr codec_ctx;
  std::optional<FilterGraph> filter_graph;

 public:
  DecoderImpl(
      const Codec<media_type>& codec,
      const std::optional<DecodeConfig>& cfg,
      const std::optional<std::string>& filter_desc);
  ~DecoderImpl() = default;

  DecoderImpl(const DecoderImpl&) = delete;
  DecoderImpl& operator=(const DecoderImpl&) = delete;
  DecoderImpl(DecoderImpl&&) = delete;
  DecoderImpl& operator=(DecoderImpl&&) = delete;

  FFmpegFramesPtr<media_type> decode(
      PacketsPtr<media_type> packets,
      int num_frames = -1);
};

} // namespace spdl::core::detail
