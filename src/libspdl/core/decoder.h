/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <MediaType media>
class DecoderImpl;
}

/// Media decoder for audio, video, or image data.
///
/// Decoder converts compressed packets into raw frames using FFmpeg codecs.
/// It supports optional filtering of decoded frames.
///
/// @tparam media The media type (Audio, Video, or Image).
template <MediaType media>
class Decoder {
  detail::DecoderImpl<media>* pImpl_;

 public:
  /// Construct a decoder.
  ///
  /// @param codec Codec information for the decoder.
  /// @param cfg Optional decode configuration.
  /// @param filter_desc Optional FFmpeg filter description to apply to decoded
  /// frames.
  Decoder(
      const Codec<media>& codec,
      const std::optional<DecodeConfig>& cfg,
      const std::optional<std::string>& filter_desc);

  /// Deleted copy constructor.
  Decoder(const Decoder&) = delete;

  /// Deleted copy assignment operator.
  Decoder& operator=(const Decoder&) = delete;

  /// Deleted move constructor.
  Decoder(Decoder&&) = delete;

  /// Deleted move assignment operator.
  Decoder& operator=(Decoder&&) = delete;

  /// Destructor.
  ~Decoder();

  /// Decode packets and flush the decoder in one operation.
  ///
  /// @param packets Packets to decode.
  /// @param num_frames Maximum number of frames to decode. Negative values
  /// decode all frames.
  /// @return Decoded frames.
  FramesPtr<media> decode_and_flush(
      PacketsPtr<media> packets,
      int num_frames = -1);

  /// Decode packets without flushing.
  ///
  /// @param packets Packets to decode.
  /// @return Decoded frames, or std::nullopt if no frames are available yet.
  std::optional<FramesPtr<media>> decode(PacketsPtr<media> packets);

  /// Flush the decoder to retrieve any remaining frames.
  ///
  /// @return Remaining decoded frames, or std::nullopt if no frames are
  /// available.
  std::optional<FramesPtr<media>> flush();
};

/// Unique pointer to a Decoder instance.
template <MediaType media>
using DecoderPtr = std::unique_ptr<Decoder<media>>;

} // namespace spdl::core
