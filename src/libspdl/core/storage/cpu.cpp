/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/storage.h>

#include "libspdl/core/detail/tracing.h"
#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#else
#include "libspdl/core/detail/logging.h"
#endif

#include <glog/logging.h>

#include <utility>

namespace spdl::core {

void* alloc_pinned(size_t s) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("`pin_memory` requires SPDL with CUDA support.");
#else
  void* p;
  CHECK_CUDA(
      cudaHostAlloc(&p, s, cudaHostAllocDefault),
      "Failed to allocate pinned memory.");
  return p;
#endif
}

void dealloc_pinned(void* p) {
#ifndef SPDL_USE_CUDA
  LOG(WARNING) << "SPDL is not compiled with CUDA support, and "
                  "`memory_pinned` attribute should not be true.";
#else
  auto status = cudaFreeHost(p);
  if (status != cudaSuccess) {
    LOG(ERROR) << fmt::format(
        "Failed to free CUDA memory ({}: {})",
        cudaGetErrorName(status),
        cudaGetErrorString(status));
  }
#endif
}

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
    : deallocator(dealloc), size(s), memory_pinned(pin_memory) {
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
  return memory_pinned;
}
CPUStorage& CPUStorage::operator=(CPUStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(deallocator, other.deallocator);
  return *this;
}
CPUStorage::~CPUStorage() {
  if (data_) {
    TRACE_EVENT(
        "decoding",
        "CPUStorage::~CPUStorage",
        perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
    deallocator(data_);
  }
}

void* CPUStorage::data() const {
  return data_;
}
} // namespace spdl::core
