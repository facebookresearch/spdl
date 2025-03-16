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

/// Abstract base buffer class (technically not needed)
/// Represents contiguous array memory.
struct Buffer {
  ///
  /// Shape of buffer
  std::vector<size_t> shape;
  ///
  /// Type of unit element
  ElemClass elem_class = ElemClass::UInt;
  ///
  /// Size of unit element
  size_t depth = sizeof(uint8_t);

  Buffer(std::vector<size_t> shape, ElemClass elem_class, size_t depth);
  virtual ~Buffer() = default;

  ///
  /// Returns the pointer to the head of the data buffer.
  virtual void* data() = 0;
};

///
/// Contiguous array data on CPU memory.
struct CPUBuffer : public Buffer {
  std::shared_ptr<CPUStorage> storage;

  CPUBuffer(
      const std::vector<size_t>& shape,
      ElemClass elem_class,
      size_t depth,
      std::shared_ptr<CPUStorage> storage);

  void* data() override;
};

using CPUBufferPtr = std::unique_ptr<CPUBuffer>;

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

/// Create ``CPUBuffer``.
CPUBufferPtr cpu_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class = ElemClass::UInt,
    size_t depth = sizeof(uint8_t),
    std::shared_ptr<CPUStorage> storage = nullptr);

} // namespace spdl::core
