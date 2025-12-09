/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/packets.h>

namespace spdl::core {
namespace detail {
class BSFImpl;
}

/// Bitstream filter for packet manipulation.
///
/// BSF (Bitstream Filter) applies transformations to compressed packets
/// without decoding/encoding. Common uses include format conversions,
/// extracting parameters, or packet manipulation.
///
/// See https://ffmpeg.org/ffmpeg-bitstream-filters.html for available filters.
///
/// @tparam media The media type (Audio, Video, or Image).
template <MediaType media>
class BSF {
  detail::BSFImpl* pImpl_;

  Rational time_base_;
  Rational frame_rate_;

 public:
  /// Construct a bitstream filter.
  ///
  /// @param codec Codec information for the filter.
  /// @param bsf Bitstream filter name or specification.
  BSF(const Codec<media>& codec, const std::string& bsf);

  /// Deleted copy constructor.
  BSF(const BSF&) = delete;

  /// Deleted copy assignment operator.
  BSF& operator=(const BSF&) = delete;

  /// Deleted move constructor.
  BSF(BSF&&) = delete;

  /// Deleted move assignment operator.
  BSF& operator=(BSF&&) = delete;

  /// Destructor.
  ~BSF();

  /// Get the codec information after filtering.
  ///
  /// @return Codec information that may be modified by the filter.
  Codec<media> get_codec() const;

  /// Apply the bitstream filter to packets.
  ///
  /// @param packets Packets to filter.
  /// @param flush Whether to flush the filter.
  /// @return Filtered packets, or std::nullopt if no packets are available yet.
  std::optional<PacketsPtr<media>> filter(
      PacketsPtr<media> packets,
      bool flush = false);

  /// Flush the filter to retrieve any remaining packets.
  ///
  /// @return Remaining filtered packets, or std::nullopt if no packets are
  /// available.
  std::optional<PacketsPtr<media>> flush();
};

} // namespace spdl::core
