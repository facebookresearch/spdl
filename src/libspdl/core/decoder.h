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
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <MediaType media_type>
class DecoderImpl;
}

template <MediaType media_type>
class Decoder {
  detail::DecoderImpl<media_type>* pImpl;

 public:
  Decoder(
      const Codec<media_type>& codec,
      const std::optional<DecodeConfig>& cfg,
      const std::optional<std::string>& filter_desc);
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
  Decoder(Decoder&&) = delete;
  Decoder& operator=(Decoder&&) = delete;

  ~Decoder();

  FFmpegFramesPtr<media_type> decode(
      PacketsPtr<media_type> packets,
      int num_frames = -1);
};

template <MediaType media_type>
using DecoderPtr = std::unique_ptr<Decoder<media_type>>;

} // namespace spdl::core
