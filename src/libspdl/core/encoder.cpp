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

template <MediaType media>
Encoder<media>::Encoder(detail::EncoderImpl<media>* p) : pImpl(p) {}

template <MediaType media>
Encoder<media>::~Encoder() {
  delete pImpl;
}

template <MediaType media>
PacketsPtr<media> Encoder<media>::encode(const FramesPtr<media>&& frames) {
  return pImpl->encode(std::move(frames));
}

template <MediaType media>
PacketsPtr<media> Encoder<media>::flush() {
  return pImpl->flush();
}

template <MediaType media>
int Encoder<media>::get_frame_size() const
  requires(media == MediaType::Audio)
{
  return pImpl->get_frame_size();
}

template class Encoder<MediaType::Audio>;
template class Encoder<MediaType::Video>;

} // namespace spdl::core
