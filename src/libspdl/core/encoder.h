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
#include <string>

namespace spdl::core {

namespace detail {
template <MediaType media_type>
class EncoderImpl;
}

template <MediaType media_type>
class Encoder {
  detail::EncoderImpl<media_type>* pImpl;

 public:
  explicit Encoder(detail::EncoderImpl<media_type>*);
  Encoder(const Encoder<media_type>&) = delete;
  Encoder& operator=(const Encoder<media_type>&) = delete;
  Encoder(Encoder<media_type>&&) = delete;
  Encoder& operator=(Encoder<media_type>&&) = delete;

  ~Encoder();

  PacketsPtr<media_type> encode(const FFmpegFramesPtr<media_type>&&);

  PacketsPtr<media_type> flush();
};

using VideoEncoder = Encoder<MediaType::Video>;
using VideoEncoderPtr = std::unique_ptr<VideoEncoder>;

} // namespace spdl::core
