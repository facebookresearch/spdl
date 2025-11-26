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
  detail::MuxerImpl* pImpl_;
  std::vector<MediaType> types_;

 public:
  Muxer(const std::string& uri, const std::optional<std::string>& muxer);

  Muxer(const Muxer&) = delete;
  Muxer& operator=(const Muxer&) = delete;
  Muxer(Muxer&&) = delete;
  Muxer& operator=(Muxer&&) = delete;

  ~Muxer();

  template <MediaType media>
  EncoderPtr<media> add_encode_stream(
      const EncodeConfigBase<media>& codec_config,
      const std::optional<std::string>& encoder,
      const std::optional<OptionDict>& encoder_config);

  template <MediaType media>
  void add_remux_stream(const Codec<media>& codec);

  void open(const std::optional<OptionDict>& muxer_config);

  template <MediaType media>
  void write(int i, Packets<media>& packets);

  void flush();
  void close();
};

using MuxerPtr = std::unique_ptr<Muxer>;

} // namespace spdl::core
