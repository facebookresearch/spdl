/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>
#include <libspdl/cuda/storage.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace spdl::cuda {
////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////

///
/// Contiguous array data on a CUDA device.
struct CUDABuffer : core::Buffer {
#ifdef SPDL_USE_CUDA
  CUDAStoragePtr storage;
  int device_index;

  CUDABuffer(
      std::vector<size_t> shape,
      core::ElemClass elem_class,
      size_t depth,
      CUDAStoragePtr storage,
      int device_index);

  void* data() override;

  uintptr_t get_cuda_stream() const;

#endif
};

using CUDABufferPtr = std::unique_ptr<CUDABuffer>;

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

// TODO: Remove this conditional
#ifdef SPDL_USE_CUDA
///
/// Create ``CUDABuffer``.
CUDABufferPtr cuda_buffer(
    const std::vector<size_t>& shape,
    const CUDAConfig& cfg,
    core::ElemClass elem_class = core::ElemClass::UInt,
    size_t depth = sizeof(uint8_t));
#endif

} // namespace spdl::cuda
