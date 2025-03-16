/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libspdl/cuda/nvdec/detail/buffer.h"

#include <cuviddec.h>

namespace spdl::core::detail {

struct Converter {
  CUstream stream;
  CUDABufferTracker* tracker;

  Converter(CUstream stream, CUDABufferTracker* tracker);

  virtual ~Converter() = default;

  /// @param src_ptr CUdeviceptr
  /// @param src_pitch Pitch in bytes
  virtual void convert(uint8_t* src_ptr, unsigned int src_pitch) = 0;
};

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABufferTracker* tracker,
    const CUVIDDECODECREATEINFO* decoder_param,
    unsigned char matrix_coeff,
    const std::optional<std::string>& pix_fmt);

} // namespace spdl::core::detail
