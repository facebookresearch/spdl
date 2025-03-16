/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>
#include <libspdl/cuda/buffer.h>

#include <optional>
#include <string>

namespace spdl::core {

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);

CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt);

namespace detail {
class NvDecDecoderInternal;
};

class NvDecDecoder {
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
      VideoPacketsPtr&& packets,
      const CUDAConfig& cuda_config,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt);
};

} // namespace spdl::core
