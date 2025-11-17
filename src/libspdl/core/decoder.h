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

#include <optional>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <MediaType media>
class DecoderImpl;
}

template <MediaType media>
class Decoder {
  detail::DecoderImpl<media>* pImpl;

 public:
  Decoder(
      const Codec<media>& codec,
      const std::optional<DecodeConfig>& cfg,
      const std::optional<std::string>& filter_desc);
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
  Decoder(Decoder&&) = delete;
  Decoder& operator=(Decoder&&) = delete;

  ~Decoder();

  FramesPtr<media> decode_and_flush(
      PacketsPtr<media> packets,
      int num_frames = -1);

  std::optional<FramesPtr<media>> decode(PacketsPtr<media> packets);
  std::optional<FramesPtr<media>> flush();
};

template <MediaType media>
using DecoderPtr = std::unique_ptr<Decoder<media>>;

} // namespace spdl::core
