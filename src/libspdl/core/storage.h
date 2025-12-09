/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <cstddef>
#include <cstdint>

namespace spdl::core {
/// Abstract base class for memory storage.
struct Storage {
  /// Get pointer to the stored data.
  ///
  /// @return Pointer to the data.
  virtual void* data() const = 0;

  /// Virtual destructor.
  virtual ~Storage() = default;
};

/// CPU memory storage with optional memory pinning.
///
/// CPUStorage manages a block of CPU memory with configurable allocation
/// and deallocation functions. Supports pinned (page-locked) memory for
/// efficient transfers to GPU.
class CPUStorage : public Storage {
  using allocator_type = std::add_pointer_t<void*(size_t)>;
  using deallocator_type = std::add_pointer_t<void(void*)>;

  static void* default_alloc(size_t s);
  static void default_dealloc(void* p);

  deallocator_type deallocator_;

 public:
  /// Size of the storage in bytes.
  size_t size;
  // So far, we only need this in CPUStorage. So we are not adding it
  // in CUDAStorage. If we need to add that to CUDAStorage, revisit
  // the interface/abstraction. (Is virtual `get_size` better?)

 private:
  void* data_ = nullptr;
  bool memory_pinned_ = false;

 public:
  /// Get pointer to the stored data.
  ///
  /// @return Pointer to the data.
  void* data() const override;

  /// Check if memory is pinned.
  ///
  /// @return true if memory is pinned, false otherwise.
  bool is_pinned() const;

  /// Default constructor.
  CPUStorage() = default;

  /// Construct CPU storage with custom allocator.
  ///
  /// @param size Size of storage in bytes.
  /// @param allocator Optional custom allocator function.
  /// @param deallocator Optional custom deallocator function.
  /// @param pin_memory Whether to pin (page-lock) the memory.
  explicit CPUStorage(
      size_t size,
      allocator_type allocator = default_alloc,
      deallocator_type deallocator = default_dealloc,
      bool pin_memory = false);

  /// Deleted copy constructor.
  CPUStorage(const CPUStorage&) = delete;

  /// Deleted copy assignment operator.
  CPUStorage& operator=(const CPUStorage&) = delete;

  /// Move constructor.
  CPUStorage(CPUStorage&&) noexcept;

  /// Move assignment operator.
  CPUStorage& operator=(CPUStorage&&) noexcept;

  /// Destructor.
  ~CPUStorage() override;
};

} // namespace spdl::core
