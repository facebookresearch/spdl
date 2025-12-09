/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/storage.h>
#include <libspdl/cuda/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace spdl::cuda {
/// Allocate pinned (page-locked) CPU memory.
///
/// Pinned memory enables faster transfers between CPU and GPU.
///
/// @param s Size in bytes to allocate.
/// @return Pointer to pinned memory.
void* alloc_pinned(size_t s);

/// Deallocate pinned CPU memory.
///
/// @param p Pointer to pinned memory to free.
void dealloc_pinned(void* p);

/// GPU memory storage.
///
/// CUDAStorage manages a block of CUDA device memory with configurable
/// allocation and deallocation functions. Supports custom allocators
/// for specialized memory management.
class CUDAStorage : public core::Storage {
  void* data_ = nullptr;

 public:
  /// CUDA stream associated with this storage.
  uintptr_t stream_ = 0;

  /// Custom deleter function for freeing memory.
  cuda_deleter_fn deleter_;

  /// Get pointer to the stored data.
  ///
  /// @return Pointer to device data.
  void* data() const override;

  /// Default constructor.
  CUDAStorage() = default;

  /// Construct CUDA storage.
  ///
  /// @param size Size of storage in bytes.
  /// @param cfg CUDA configuration including device, stream, and optional
  /// allocator.
  CUDAStorage(size_t size, const CUDAConfig& cfg);

  /// Deleted copy constructor.
  CUDAStorage(const CUDAStorage&) = delete;

  /// Deleted copy assignment operator.
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  /// Move constructor.
  CUDAStorage(CUDAStorage&&) noexcept;

  /// Move assignment operator.
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  /// Destructor.
  ~CUDAStorage() override;
};

/// Shared pointer to a CUDAStorage.
using CUDAStoragePtr = std::shared_ptr<CUDAStorage>;

} // namespace spdl::cuda
