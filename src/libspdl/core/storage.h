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
#include <functional>
#include <optional>

namespace spdl::core {
void* alloc_pinned(size_t s);
void dealloc_pinned(void* p);

struct Storage {
  virtual void* data() const = 0;
  virtual ~Storage() = default;
};

class CPUStorage : public Storage {
  using allocator_type = std::add_pointer_t<void*(size_t)>;
  using deallocator_type = std::add_pointer_t<void(void*)>;

  static void* default_alloc(size_t s);
  static void default_dealloc(void* p);

  deallocator_type deallocator;

 public:
  size_t size;
  // So far, we only need this in CPUStorage. So we are not adding it
  // in CUDAStorage. If we need to add that to CUDAStorage, revisit
  // the interface/abstraction. (Is virtual `get_size` better?)
 private:
  void* data_ = nullptr;
  bool memory_pinned = false;

 public:
  void* data() const override;
  bool is_pinned() const;

  CPUStorage() = default;
  explicit CPUStorage(
      size_t size,
      allocator_type = default_alloc,
      deallocator_type = default_dealloc,
      bool pin_memory = false);

  CPUStorage(const CPUStorage&) = delete;
  CPUStorage& operator=(const CPUStorage&) = delete;

  CPUStorage(CPUStorage&&) noexcept;
  CPUStorage& operator=(CPUStorage&&) noexcept;

  ~CPUStorage() override;
};

} // namespace spdl::core
