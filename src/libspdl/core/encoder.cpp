/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/encoder.h>

#include <libspdl/core/detail/ffmpeg/encoder.h>

namespace spdl::core {

template <MediaType media_type>
Encoder<media_type>::Encoder(detail::EncoderImpl<media_type>* p) : pImpl(p) {}

template <MediaType media_type>
Encoder<media_type>::~Encoder() {
  delete pImpl;
}

template <MediaType media_type>
PacketsPtr<media_type> Encoder<media_type>::encode(
    const FFmpegFramesPtr<media_type>&& frames) {
  return pImpl->encode(std::move(frames));
}

template <MediaType media_type>
PacketsPtr<media_type> Encoder<media_type>::flush() {
  return pImpl->flush();
}

template class Encoder<MediaType::Audio>;
template class Encoder<MediaType::Video>;

} // namespace spdl::core
