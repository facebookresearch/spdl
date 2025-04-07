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

template <MediaType media>
class BSF {
  detail::BSFImpl* pImpl;

  Rational time_base;
  Rational frame_rate;

 public:
  BSF(const Codec<media>& codec, const std::string& bsf);
  BSF(const BSF&) = delete;
  BSF& operator=(const BSF&) = delete;
  BSF(BSF&&) = delete;
  BSF& operator=(BSF&&) = delete;
  ~BSF();

  Codec<media> get_codec() const;

  PacketsPtr<media> filter(PacketsPtr<media> packets, bool flush = false);
  PacketsPtr<media> flush();
};

} // namespace spdl::core
