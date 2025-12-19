/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/muxer.h>

#include "libspdl/common/logging.h"
#include "libspdl/core/detail/ffmpeg/muxer.h"

#include <fmt/core.h>

namespace spdl::core {

Muxer::Muxer(const std::string& uri, const std::optional<std::string>& muxer)
    : pImpl_(new detail::MuxerImpl(uri, muxer)) {}

Muxer::~Muxer() {
  delete pImpl_;
}

template <MediaType media>
EncoderPtr<media> Muxer::add_encode_stream(
    const EncodeConfigBase<media>& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config) {
  auto p = pImpl_->add_encode_stream(codec_config, encoder, encoder_config);
  types_.push_back(media);
  return std::make_unique<Encoder<media>>(p.release());
}

template AudioEncoderPtr Muxer::add_encode_stream(
    const AudioEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);
template VideoEncoderPtr Muxer::add_encode_stream(
    const VideoEncodeConfig& codec_config,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_config);

template <MediaType media>
void Muxer::add_remux_stream(const Codec<media>& codec) {
  pImpl_->add_remux_stream(codec);
  types_.push_back(media);
}

template void Muxer::add_remux_stream(const AudioCodec& codec);
template void Muxer::add_remux_stream(const VideoCodec& codec);

void Muxer::open(const std::optional<OptionDict>& muxer_config) {
  pImpl_->open(muxer_config);
}

namespace {
std::string to_str(MediaType media) {
  switch (media) {
    case MediaType::Audio:
      return "audio";
    case MediaType::Video:
      return "video";
    case MediaType::Image:
      return "image";
    default:
      throw std::domain_error("[INTERNAL ERROR] Unexpected media type.");
  }
}
} // namespace

template <MediaType media>
void Muxer::write(int i, Packets<media>& packets) {
  if (i < 0 || i >= (int)types_.size()) {
    SPDL_FAIL(
        fmt::format("Index {} is out of range (0, {}]", i, types_.size()));
  }
  if (types_.at(i) != media) {
    SPDL_FAIL(
        fmt::format(
            "Stream {} expects {} type, but {} type was provided.",
            i,
            to_str(types_.at(i)),
            to_str(media)));
  }
  pImpl_->write(i, packets.pkts.get_packets(), packets.time_base);
}

template void Muxer::write<MediaType::Audio>(int, Packets<MediaType::Audio>&);
template void Muxer::write<MediaType::Video>(int, Packets<MediaType::Video>&);

void Muxer::flush() {
  pImpl_->flush();
}

void Muxer::close() {
  pImpl_->close();
}

} // namespace spdl::core
