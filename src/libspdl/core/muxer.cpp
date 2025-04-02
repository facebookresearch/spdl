/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/muxer.h>

#include "libspdl/core/detail/ffmpeg/muxer.h"

namespace spdl::core {

Muxer::Muxer(const std::string& uri, const std::optional<std::string>& muxer)
    : pImpl(new detail::MuxerImpl(uri, muxer)) {}

Muxer::~Muxer() {
  delete pImpl;
}

VideoEncoderPtr Muxer::add_video_encode_stream(
    const VideoEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config) {
  auto p = pImpl->add_encode_stream(codec_config, encoder, encoder_config);
  return std::make_unique<VideoEncoder>(p.release());
}

void Muxer::add_video_remux_stream(const VideoCodec& codec) {
  pImpl->add_remux_stream(codec);
}

void Muxer::open(const std::optional<OptionDict>& muxer_config) {
  pImpl->open(muxer_config);
}

template <MediaType media_type>
void Muxer::write(int i, DemuxedPackets<media_type>& packets) {
  pImpl->write(i, packets.get_packets(), packets.codec.get_time_base());
}

template void Muxer::write<MediaType::Video>(
    int,
    DemuxedPackets<MediaType::Video>&);

void Muxer::flush() {
  pImpl->flush();
}

void Muxer::close() {
  pImpl->close();
}

} // namespace spdl::core
