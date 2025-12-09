/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>
#include <libspdl/cuda/storage.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace spdl::cuda {
////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////

/// Contiguous array data on a CUDA device.
///
/// CUDABuffer represents a multi-dimensional array in CUDA device memory
/// with shape and data type information. Used for GPU-accelerated
/// media processing.
struct CUDABuffer {
  /// CUDA device index.
  int device_index;
  /// Storage backing the buffer data on GPU.
  CUDAStoragePtr storage;

  /// Shape of the buffer (dimensions).
  std::vector<size_t> shape;

  /// Element class (Int, UInt, or Float).
  core::ElemClass elem_class = core::ElemClass::UInt;

  /// Size of each element in bytes.
  size_t depth = sizeof(uint8_t);

  /// Get pointer to the buffer data.
  ///
  /// @return Pointer to device data.
  void* data();

  /// Get pointer to the buffer data (const version).
  ///
  /// @return Pointer to device data.
  void* data() const;

  /// Get the CUDA stream associated with this buffer.
  ///
  /// @return CUDA stream handle.
  uintptr_t get_cuda_stream() const;
};

/// Unique pointer to a CUDABuffer.
using CUDABufferPtr = std::unique_ptr<CUDABuffer>;

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

/// Create a CUDA buffer with specified shape and data type.
///
/// @param shape Dimensions of the buffer.
/// @param cfg CUDA configuration including device index and stream.
/// @param elem_class Element class (Int, UInt, or Float).
/// @param depth Size of each element in bytes.
/// @return CUDABuffer instance.
CUDABufferPtr cuda_buffer(
    const std::vector<size_t>& shape,
    const CUDAConfig& cfg,
    spdl::core::ElemClass elem_class = spdl::core::ElemClass::UInt,
    size_t depth = sizeof(uint8_t));

} // namespace spdl::cuda
