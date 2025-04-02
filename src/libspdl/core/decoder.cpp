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
    : pImpl(new detail::DecoderImpl<media>(codec, cfg, filter_desc)) {}

template <MediaType media>
Decoder<media>::~Decoder() {
  delete pImpl;
}

template <MediaType media>
FramesPtr<media> Decoder<media>::decode_and_flush(
    PacketsPtr<media> packets,
    int num_frames) {
  return pImpl->decode_and_flush(std::move(packets), num_frames);
}

template <MediaType media>
FramesPtr<media> Decoder<media>::decode(PacketsPtr<media> packets) {
  return pImpl->decode(std::move(packets));
}

template <MediaType media>
FramesPtr<media> Decoder<media>::flush() {
  return pImpl->flush();
}

template class Decoder<MediaType::Audio>;
template class Decoder<MediaType::Video>;
template class Decoder<MediaType::Image>;

} // namespace spdl::core
