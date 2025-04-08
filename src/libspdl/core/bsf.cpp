/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/bsf.h>
#include <libspdl/core/codec.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"

namespace spdl::core {

template <MediaType media>
BSF<media>::BSF(const Codec<media>& codec, const std::string& bsf)
    : pImpl(new detail::BSFImpl(bsf, codec.get_parameters())),
      time_base(codec.get_time_base()),
      frame_rate(codec.get_frame_rate()) {}

template <MediaType media>
BSF<media>::~BSF() {
  delete pImpl;
}

template <MediaType media>
Codec<media> BSF<media>::get_codec() const {
  auto* codecpar = pImpl->get_output_codec_par();
  return Codec<media>{codecpar, time_base, frame_rate};
}

template <MediaType media>
PacketsPtr<media> BSF<media>::filter(PacketsPtr<media> packets, bool flush) {
  auto ret = std::make_unique<Packets<media>>(
      packets->src, packets->stream_index, time_base, packets->timestamp);
  if (packets->codec) {
    ret->codec = get_codec();
  }
  pImpl->filter(packets->pkts.get_packets(), ret->pkts, flush);
  return ret;
}

template <MediaType media>
PacketsPtr<media> BSF<media>::flush() {
  auto ret = std::make_unique<Packets<media>>("0", -1, time_base);
  pImpl->flush(ret->pkts);
  return ret;
}

template class BSF<MediaType::Video>;
template class BSF<MediaType::Audio>;
template class BSF<MediaType::Image>;

} // namespace spdl::core
