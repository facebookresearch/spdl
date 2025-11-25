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
template <MediaType media>
class DecoderImpl {
  AVCodecContextPtr codec_ctx_;
  std::optional<FilterGraphImpl> filter_graph_;

 public:
  DecoderImpl(
      const Codec<media>& codec,
      const std::optional<DecodeConfig>& cfg,
      const std::optional<std::string>& filter_desc);
  ~DecoderImpl() = default;

  DecoderImpl(const DecoderImpl&) = delete;
  DecoderImpl& operator=(const DecoderImpl&) = delete;
  DecoderImpl(DecoderImpl&&) = delete;
  DecoderImpl& operator=(DecoderImpl&&) = delete;

  Rational get_output_time_base() const;

  FramesPtr<media> decode_and_flush(
      PacketsPtr<media> packets,
      int num_frames = -1);
  FramesPtr<media> decode(PacketsPtr<media> packets);
  FramesPtr<media> flush();
};

} // namespace spdl::core::detail
