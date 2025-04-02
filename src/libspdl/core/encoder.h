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

  PacketsPtr<media_type> encode(const FramesPtr<media_type>&&);

  PacketsPtr<media_type> flush();
};

template <MediaType media_type>
using EncoderPtr = std::unique_ptr<Encoder<media_type>>;

using VideoEncoder = Encoder<MediaType::Video>;
using VideoEncoderPtr = EncoderPtr<MediaType::Video>;

using AudioEncoder = Encoder<MediaType::Audio>;
using AudioEncoderPtr = EncoderPtr<MediaType::Audio>;

} // namespace spdl::core
