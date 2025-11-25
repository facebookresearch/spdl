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

template <MediaType media>
class Encoder {
  detail::EncoderImpl<media>* pImpl_;

 public:
  explicit Encoder(detail::EncoderImpl<media>*);
  Encoder(const Encoder<media>&) = delete;
  Encoder& operator=(const Encoder<media>&) = delete;
  Encoder(Encoder<media>&&) = delete;
  Encoder& operator=(Encoder<media>&&) = delete;

  ~Encoder();

  std::optional<PacketsPtr<media>> encode(const FramesPtr<media>&&);

  std::optional<PacketsPtr<media>> flush();

  int get_frame_size() const
    requires(media == MediaType::Audio);
};

template <MediaType media>
using EncoderPtr = std::unique_ptr<Encoder<media>>;

using VideoEncoder = Encoder<MediaType::Video>;
using VideoEncoderPtr = EncoderPtr<MediaType::Video>;

using AudioEncoder = Encoder<MediaType::Audio>;
using AudioEncoderPtr = EncoderPtr<MediaType::Audio>;

} // namespace spdl::core
