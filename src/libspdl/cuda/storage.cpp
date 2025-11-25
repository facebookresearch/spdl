/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/storage.h>

#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/detail/utils.h"

#include <fmt/core.h>
#include <glog/logging.h>

namespace spdl::cuda {
////////////////////////////////////////////////////////////////////////////////
// Pinned allocator for CPUStorage
////////////////////////////////////////////////////////////////////////////////

void* alloc_pinned(size_t s) {
  void* p;
  CHECK_CUDA(
      cudaHostAlloc(&p, s, cudaHostAllocDefault),
      "Failed to allocate pinned memory.");
  return p;
}

void dealloc_pinned(void* p) {
  auto status = cudaFreeHost(p);
  if (status != cudaSuccess) {
    LOG(ERROR) << fmt::format(
        "Failed to free CUDA memory ({}: {})",
        cudaGetErrorName(status),
        cudaGetErrorString(status));
  }
}

////////////////////////////////////////////////////////////////////////////////
// CUDAStorage
////////////////////////////////////////////////////////////////////////////////
namespace {
uintptr_t default_allocator(int size, int device, uintptr_t) {
  {
    TRACE_EVENT("decoding", "cudaSetDevice");
    CHECK_CUDA(cudaSetDevice(device), "Failed to set current context.");
  }
  void* data = nullptr;
  {
    TRACE_EVENT("decoding", "cudaMalloc");
    CHECK_CUDA(cudaMalloc(&data, size), "Failed to allocate CUDA memory");
  }
  return reinterpret_cast<uintptr_t>(data);
}

void default_deleter(uintptr_t data) {
  auto status = cudaFree((void*)data);
  if (status != cudaSuccess) {
    LOG(ERROR) << fmt::format(
        "Failed to free CUDA memory ({}: {})",
        cudaGetErrorName(status),
        cudaGetErrorString(status));
  }
}

cuda_allocator default_alloc = {default_allocator, default_deleter};

} // namespace

CUDAStorage::CUDAStorage(size_t size, const CUDAConfig& cfg)
    : stream_(cfg.stream) {
  if (size == 0) {
    SPDL_FAIL("`size` must be greater than 0.");
  }

  TRACE_EVENT("decoding", "custom_cuda_allocator_fn");
  auto [allocator_fn, deleter_fn] = cfg.allocator.value_or(default_alloc);
  data_ =
      reinterpret_cast<void*>(allocator_fn(size, cfg.device_index, cfg.stream));
  deleter_ = std::move(deleter_fn);
}

CUDAStorage::CUDAStorage(CUDAStorage&& other) noexcept {
  *this = std::move(other);
}

CUDAStorage& CUDAStorage::operator=(CUDAStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(stream_, other.stream_);
  swap(deleter_, other.deleter_);
  return *this;
}
CUDAStorage::~CUDAStorage() {
  if (data_) {
    TRACE_EVENT("decoding", "CUDAStorage::~CUDAStorage");
    VLOG(9) << "Freeing CUDA memory " << data_;
    deleter_(reinterpret_cast<uintptr_t>(data_));
  }
}

void* CUDAStorage::data() const {
  return data_;
}

} // namespace spdl::cuda
