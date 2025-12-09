/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/storage.h>
#include <libspdl/core/types.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////

/// Contiguous array data on CPU memory.
///
/// CPUBuffer represents a multi-dimensional array in CPU memory with
/// shape and data type information. Used for exposing the decoded media frames
/// data to array libraries like NumPy and PyTorch.
struct CPUBuffer {
  /// Storage backing the buffer data.
  std::shared_ptr<CPUStorage> storage;

  /// Shape of the buffer (dimensions).
  std::vector<size_t> shape;

  /// Element class (Int, UInt, or Float).
  ElemClass elem_class = ElemClass::UInt;

  /// Size of each element in bytes.
  size_t depth = sizeof(uint8_t);

  /// Get pointer to the buffer data.
  ///
  /// @return Pointer to the data.
  void* data();
};

/// Unique pointer to a CPUBuffer.
using CPUBufferPtr = std::unique_ptr<CPUBuffer>;

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

/// Create a CPU buffer with specified shape and data type.
///
/// @param shape Dimensions of the buffer.
/// @param elem_class Element class (Int, UInt, or Float).
/// @param depth Size of each element in bytes.
/// @param storage Optional pre-allocated storage. If not provided, allocates
/// new storage.
/// @return CPUBuffer instance.
CPUBufferPtr cpu_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class = ElemClass::UInt,
    size_t depth = sizeof(uint8_t),
    std::shared_ptr<CPUStorage> storage = nullptr);

} // namespace spdl::core
