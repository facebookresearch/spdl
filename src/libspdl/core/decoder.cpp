/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoder.h>

#include "libspdl/core/detail/ffmpeg/decoder.h"

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
FramesPtr<media> Decoder<media>::decode_packets(
    PacketsPtr<media> packets,
    int num_frames) {
  return pImpl_->decode_packets(std::move(packets), num_frames);
}

template <MediaType media>
Generator<FramesPtr<media>> Decoder<media>::streaming_decode_packets(
    PacketsPtr<media> packets)
  requires(media == MediaType::Video || media == MediaType::Audio)
{
  return pImpl_->streaming_decode_packets(std::move(packets));
}

template <MediaType media>
Generator<FramesPtr<media>> Decoder<media>::flush()
  requires(media == MediaType::Video || media == MediaType::Audio)
{
  return pImpl_->flush();
}

template class Decoder<MediaType::Audio>;
template class Decoder<MediaType::Video>;
template class Decoder<MediaType::Image>;

} // namespace spdl::core
