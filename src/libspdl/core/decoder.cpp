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
template <MediaType media_type>
Decoder<media_type>::Decoder(
    const Codec<media_type>& codec,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc)
    : pImpl(new detail::DecoderImpl<media_type>(codec, cfg, filter_desc)) {}

template <MediaType media_type>
Decoder<media_type>::~Decoder() {
  delete pImpl;
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> Decoder<media_type>::decode_and_flush(
    PacketsPtr<media_type> packets,
    int num_frames) {
  return pImpl->decode_and_flush(std::move(packets), num_frames);
}

template class Decoder<MediaType::Audio>;
template class Decoder<MediaType::Video>;
template class Decoder<MediaType::Image>;

} // namespace spdl::core
