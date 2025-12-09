/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>

namespace spdl::core {

namespace detail {
template <MediaType media>
class EncoderImpl;
}

/// Media encoder for compressing raw frames into packets.
///
/// Encoder converts raw frames (audio or video) into compressed packets
/// using FFmpeg codecs.
///
/// @tparam media The media type (Audio or Video).
template <MediaType media>
class Encoder {
  detail::EncoderImpl<media>* pImpl_;

 public:
  /// Construct an encoder from implementation.
  ///
  /// @param pImpl Pointer to encoder implementation.
  explicit Encoder(detail::EncoderImpl<media>* pImpl);

  /// Deleted copy constructor.
  Encoder(const Encoder<media>&) = delete;

  /// Deleted copy assignment operator.
  Encoder& operator=(const Encoder<media>&) = delete;

  /// Deleted move constructor.
  Encoder(Encoder<media>&&) = delete;

  /// Deleted move assignment operator.
  Encoder& operator=(Encoder<media>&&) = delete;

  /// Destructor.
  ~Encoder();

  /// Encode frames into packets.
  ///
  /// @param frames Frames to encode.
  /// @return Encoded packets, or std::nullopt if no packets are available yet.
  std::optional<PacketsPtr<media>> encode(const FramesPtr<media>&&);

  /// Flush the encoder to retrieve any remaining packets.
  ///
  /// @return Remaining encoded packets, or std::nullopt if no packets are
  /// available.
  std::optional<PacketsPtr<media>> flush();

  /// Get the frame size for audio encoding.
  ///
  /// @return Number of samples per frame required by the encoder.
  int get_frame_size() const
    requires(media == MediaType::Audio);
};

/// Unique pointer to an Encoder instance.
template <MediaType media>
using EncoderPtr = std::unique_ptr<Encoder<media>>;

/// Video encoder type alias.
using VideoEncoder = Encoder<MediaType::Video>;
/// Unique pointer to a VideoEncoder instance.
using VideoEncoderPtr = EncoderPtr<MediaType::Video>;

/// Audio encoder type alias.
using AudioEncoder = Encoder<MediaType::Audio>;
/// Unique pointer to an AudioEncoder instance.
using AudioEncoderPtr = EncoderPtr<MediaType::Audio>;

} // namespace spdl::core
