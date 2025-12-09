/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/encoder.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string>

namespace spdl::core {
namespace detail {
class MuxerImpl;
} // namespace detail

/// Media muxer for writing packets to container formats.
///
/// Muxer writes compressed packets to output files or streams, combining
/// multiple streams (audio, video, etc.) into a single container format.
class Muxer {
  detail::MuxerImpl* pImpl_;
  std::vector<MediaType> types_;

 public:
  /// Construct a muxer for writing to a URI.
  ///
  /// @param uri Output URI (file path, etc.).
  /// @param muxer Optional muxer format name.
  Muxer(const std::string& uri, const std::optional<std::string>& muxer);

  /// Deleted copy constructor.
  Muxer(const Muxer&) = delete;

  /// Deleted copy assignment operator.
  Muxer& operator=(const Muxer&) = delete;

  /// Deleted move constructor.
  Muxer(Muxer&&) = delete;

  /// Deleted move assignment operator.
  Muxer& operator=(Muxer&&) = delete;

  /// Destructor.
  ~Muxer();

  /// Add a stream with encoding.
  ///
  /// @tparam media Media type (Audio or Video).
  /// @param codec_config Encoding configuration.
  /// @param encoder Optional encoder name.
  /// @param encoder_config Optional encoder-specific options.
  /// @return Encoder instance for the new stream.
  template <MediaType media>
  EncoderPtr<media> add_encode_stream(
      const EncodeConfigBase<media>& codec_config,
      const std::optional<std::string>& encoder,
      const std::optional<OptionDict>& encoder_config);

  /// Add a stream for remuxing (without re-encoding).
  ///
  /// @tparam media Media type (Audio or Video).
  /// @param codec Codec information from the source.
  template <MediaType media>
  void add_remux_stream(const Codec<media>& codec);

  /// Open the muxer for writing.
  ///
  /// @param muxer_config Optional muxer-specific options.
  void open(const std::optional<OptionDict>& muxer_config);

  /// Write packets to a stream.
  ///
  /// @tparam media Media type (Audio or Video).
  /// @param i Stream index.
  /// @param packets Packets to write.
  template <MediaType media>
  void write(int i, Packets<media>& packets);

  /// Flush any buffered packets.
  void flush();

  /// Close the muxer and finalize the output.
  void close();
};

/// Unique pointer to a Muxer instance.
using MuxerPtr = std::unique_ptr<Muxer>;

} // namespace spdl::core
