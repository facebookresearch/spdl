/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/storage.h>

#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

namespace spdl::cuda {

namespace detail {
class NvDecDecoderCore;
}

// Usage:
// NvDecDecoder decoder{};
// decoder.init(...)
// for (auto packets : src) {
//   frames = decoder.decode(std::move(packets));
//   ...
// }
// frames = decoder.flush();
//
// If there was an error, then reset before
// initializing it again.
//
// decoder.reset();
// decoder.init();
class NvDecDecoder {
  detail::NvDecDecoderCore* core;

 public:
  NvDecDecoder();
  NvDecDecoder(const NvDecDecoder&) = delete;
  NvDecDecoder& operator=(const NvDecDecoder&) = delete;
  // Deleting the move constructor for now.
  NvDecDecoder(NvDecDecoder&&) = delete;
  NvDecDecoder& operator=(NvDecDecoder&&) = delete;

  ~NvDecDecoder();

  void reset();

  // Initialize the decoder object for a new stream of
  // video packets
  void init(
      // device config
      const CUDAConfig& cuda_config,
      // Source codec information
      const spdl::core::VideoCodec& codec,
      // Post-decoding processing params
      CropArea crop,
      int width = -1,
      int height = -1);

  // decode one stream of video packets
  std::vector<CUDABuffer> decode(spdl::core::VideoPacketsPtr packets);

  // Call this method at the end of video stream.
  std::vector<CUDABuffer> flush();
};

} // namespace spdl::cuda
