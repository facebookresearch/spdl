/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

class BSFImpl {
  AVBSFContextPtr bsf_ctx;

 public:
  BSFImpl(const std::string& name, const AVCodecParameters* codec_par);

  Generator<AVPacketPtr> filter(AVPacket* packet);
  void filter(
      const std::vector<AVPacket*>& packets,
      PacketSeries& out,
      bool flush = false);

  AVCodecParameters* get_output_codec_par();
};

} // namespace spdl::core::detail
