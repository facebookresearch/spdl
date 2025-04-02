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

class Muxer {
  detail::MuxerImpl* pImpl;

 public:
  Muxer(const std::string& uri, const std::optional<std::string>& muxer);

  Muxer(const Muxer&) = delete;
  Muxer& operator=(const Muxer&) = delete;
  Muxer(Muxer&&) = delete;
  Muxer& operator=(Muxer&&) = delete;

  ~Muxer();

  template <MediaType media_type>
  std::unique_ptr<Encoder<media_type>> add_encode_stream(
      const EncodeConfigBase<media_type>& codec_config,
      const std::optional<std::string>& encoder,
      const std::optional<OptionDict>& encoder_config);

  template <MediaType media_type>
  void add_remux_stream(const Codec<media_type>& codec);

  void open(const std::optional<OptionDict>& muxer_config);

  template <MediaType media_type>
  void write(int i, DemuxedPackets<media_type>& packets);

  void flush();
  void close();
};

using MuxerPtr = std::unique_ptr<Muxer>;

} // namespace spdl::core
