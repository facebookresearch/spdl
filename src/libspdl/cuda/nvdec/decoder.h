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

#include <optional>
#include <string>

namespace spdl::cuda {

namespace detail {
class NvDecDecoderInternal;
};

class NvDecDecoder {
#ifdef SPDL_USE_NVCODEC
  bool init = false;

  detail::NvDecDecoderInternal* decoder;

 public:
  NvDecDecoder();
  NvDecDecoder(const NvDecDecoder&) = delete;
  NvDecDecoder& operator=(const NvDecDecoder&) = delete;
  // Deleting the move constructor for now.
  NvDecDecoder(NvDecDecoder&&) = delete;
  NvDecDecoder& operator=(NvDecDecoder&&) = delete;

  ~NvDecDecoder();

  void reset();

  void set_init_flag();

  CUDABufferPtr decode(
      spdl::core::VideoPacketsPtr&& packets,
      const CUDAConfig& cuda_config,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt,
      bool flush);
#endif
};

namespace detail {
class NvDecDecoderCore2;
} // namespace detail

// Usage:
// NvDecDecoder2 decoder{};
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
class NvDecDecoder2 {
#ifdef SPDL_USE_NVCODEC
  detail::NvDecDecoderCore2* core;

 public:
  NvDecDecoder2();
  NvDecDecoder2(const NvDecDecoder2&) = delete;
  NvDecDecoder2& operator=(const NvDecDecoder2&) = delete;
  // Deleting the move constructor for now.
  NvDecDecoder2(NvDecDecoder2&&) = delete;
  NvDecDecoder2& operator=(NvDecDecoder2&&) = delete;

  ~NvDecDecoder2();

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
      int target_width = -1,
      int target_height = -1);

  // decode one stream of video packets
  std::vector<CUDABuffer> decode(spdl::core::VideoPacketsPtr packets);

  // Call this method at the end of video stream.
  std::vector<CUDABuffer> flush();
#endif
};

} // namespace spdl::cuda
