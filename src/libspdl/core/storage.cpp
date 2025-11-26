/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/storage.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <utility>

namespace spdl::core {

void* CPUStorage::default_alloc(size_t s) {
  return operator new(s);
}

void CPUStorage::default_dealloc(void* p) {
  operator delete(p);
}

////////////////////////////////////////////////////////////////////////////////
// Storage
////////////////////////////////////////////////////////////////////////////////
CPUStorage::CPUStorage(
    size_t s,
    allocator_type alloc,
    deallocator_type dealloc,
    bool pin_memory)
    : deallocator_(dealloc), size(s), memory_pinned_(pin_memory) {
  TRACE_EVENT(
      "decoding",
      "CPUStorage::CPUStorage",
      perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));

  if (size == 0) {
    SPDL_FAIL("`size` must be greater than 0.");
  }
  data_ = alloc(size);
}
CPUStorage::CPUStorage(CPUStorage&& other) noexcept {
  *this = std::move(other);
}
bool CPUStorage::is_pinned() const {
  return memory_pinned_;
}
CPUStorage& CPUStorage::operator=(CPUStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(deallocator_, other.deallocator_);
  return *this;
}
CPUStorage::~CPUStorage() {
  if (data_) {
    TRACE_EVENT(
        "decoding",
        "CPUStorage::~CPUStorage",
        perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
    deallocator_(data_);
  }
}

void* CPUStorage::data() const {
  return data_;
}
} // namespace spdl::core
