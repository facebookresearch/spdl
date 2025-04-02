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

template <MediaType media_type>
std::unique_ptr<Encoder<media_type>> Muxer::add_encode_stream(
    const EncodeConfigBase<media_type>& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config) {
  auto p = pImpl->add_encode_stream(codec_config, encoder, encoder_config);
  return std::make_unique<Encoder<media_type>>(p.release());
}

template std::unique_ptr<AudioEncoder> Muxer::add_encode_stream(
    const AudioEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);
template std::unique_ptr<VideoEncoder> Muxer::add_encode_stream(
    const VideoEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);

template <MediaType media_type>
void Muxer::add_remux_stream(const Codec<media_type>& codec) {
  pImpl->add_remux_stream(codec);
}

template void Muxer::add_remux_stream(const AudioCodec& codec);
template void Muxer::add_remux_stream(const VideoCodec& codec);

void Muxer::open(const std::optional<OptionDict>& muxer_config) {
  pImpl->open(muxer_config);
}

template <MediaType media_type>
void Muxer::write(int i, Packets<media_type>& packets) {
  pImpl->write(i, packets.get_packets(), packets.codec.get_time_base());
}

template void Muxer::write<MediaType::Audio>(int, Packets<MediaType::Audio>&);
template void Muxer::write<MediaType::Video>(int, Packets<MediaType::Video>&);

void Muxer::flush() {
  pImpl->flush();
}

void Muxer::close() {
  pImpl->close();
}

} // namespace spdl::core
