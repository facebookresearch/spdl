/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/encoder.h>

#include <libspdl/core/detail/ffmpeg/encoder.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////

VideoEncoder::VideoEncoder(detail::EncoderImpl<MediaType::Video>* p)
    : pImpl(p) {}

VideoEncoder::~VideoEncoder() {
  delete pImpl;
}

VideoPacketsPtr VideoEncoder::encode(
    const FFmpegFramesPtr<MediaType::Video>&& frames) {
  return pImpl->encode(std::move(frames));
}

VideoPacketsPtr VideoEncoder::flush() {
  return pImpl->flush();
}

} // namespace spdl::core
