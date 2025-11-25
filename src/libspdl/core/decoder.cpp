/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoder.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/decoder.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
Decoder<media>::Decoder(
    const Codec<media>& codec,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc)
    : pImpl_(new detail::DecoderImpl<media>(codec, cfg, filter_desc)) {}

template <MediaType media>
Decoder<media>::~Decoder() {
  delete pImpl_;
}

template <MediaType media>
FramesPtr<media> Decoder<media>::decode_and_flush(
    PacketsPtr<media> packets,
    int num_frames) {
  return pImpl_->decode_and_flush(std::move(packets), num_frames);
}

template <MediaType media>
std::optional<FramesPtr<media>> Decoder<media>::decode(
    PacketsPtr<media> packets) {
  auto frames = pImpl_->decode(std::move(packets));
  if (frames->get_frames().size() == 0) {
    return std::nullopt;
  }
  return frames;
}

template <MediaType media>
std::optional<FramesPtr<media>> Decoder<media>::flush() {
  auto frames = pImpl_->flush();
  if (frames->get_frames().size() == 0) {
    return std::nullopt;
  }
  return frames;
}

template class Decoder<MediaType::Audio>;
template class Decoder<MediaType::Video>;
template class Decoder<MediaType::Image>;

} // namespace spdl::core
