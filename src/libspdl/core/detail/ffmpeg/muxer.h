/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/encoder.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <optional>
#include <string>

namespace spdl::core::detail {

class MuxerImpl {
  AVFormatOutputContextPtr fmt_ctx;

 public:
  MuxerImpl(const std::string& uri, const std::optional<std::string>& format);

 private:
  template <MediaType media_type>
  void assert_media_type_is_supported() const;

 public:
  template <MediaType media_type>
  std::unique_ptr<EncoderImpl<media_type>> add_encode_stream(
      const EncodeConfigBase<media_type>& codec_config,
      const std::optional<std::string>& encoder,
      const std::optional<OptionDict>& encoder_config);

  template <MediaType media_type>
  void add_remux_stream(const Codec<media_type>& codec);

  void open(const std::optional<OptionDict>& muxer_config);

  void
  write(int i, const std::vector<AVPacket*>& packets, AVRational time_base);

  void flush();
  void close();
};

} // namespace spdl::core::detail
