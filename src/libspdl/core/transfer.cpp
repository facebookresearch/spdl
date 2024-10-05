/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/transfer.h>

#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#endif
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <mutex>

namespace spdl::core {
namespace {
size_t prod(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto& v : shape) {
    ret *= v;
  }
  return ret;
}

std::once_flag WARN_DEFAULT_STREAM_FLAG;
void warn_default_stream() noexcept {
  LOG(WARNING)
      << "The CPUStorage is page-locked (pinned), but the default CUDA stream is used. "
         "This is likely not what you intend. "
         "Please use a non-default CUDA stream to overlap the data transfer with kernel execution.";
}

CUDABufferPtr transfer_buffer_impl(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg,
    bool pinned_memory = false) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  TRACE_EVENT("decoding", "core::transfer_buffer");

  auto ret = cuda_buffer(shape, cfg, elem_class, depth);
  size_t size = depth * prod(shape);

  if (pinned_memory) {
    if (!cfg.stream) {
      std::call_once(WARN_DEFAULT_STREAM_FLAG, warn_default_stream);
    }
    auto s = (cudaStream_t)cfg.stream;
    CHECK_CUDA(
        cudaMemcpyAsync(ret->data(), ptr, size, cudaMemcpyHostToDevice, s),
        "Failed to initialite async memory copy from host to device.");
    CHECK_CUDA(cudaStreamSynchronize(s), "Failed to synchronize the stream.");

  } else {
    CHECK_CUDA(
        cudaMemcpy(ret->data(), ptr, size, cudaMemcpyHostToDevice),
        "Failed to copy data from host to device.");
  }

  return ret;
#endif
}

} // namespace

CUDABufferPtr transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg) {
  return transfer_buffer_impl(
      buffer->shape,
      buffer->elem_class,
      buffer->depth,
      buffer->data(),
      cfg,
      buffer->storage->is_pinned());
}

CUDABufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg) {
  return transfer_buffer_impl(shape, elem_class, depth, ptr, cfg);
}

CPUBufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    const void* ptr) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  TRACE_EVENT("decoding", "core::transfer_buffer");

  auto ret = cpu_buffer(shape, elem_class, depth);
  size_t size = depth * prod(shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), ptr, size, cudaMemcpyDeviceToHost),
      "Failed to copy data from device to host.");

  return ret;
#endif
}

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t>& shape) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else

  size_t size = prod(shape);
  CPUStorage storage{size};

  CHECK_CUDA(
      cudaMemcpy(storage.data(), src, size, cudaMemcpyDeviceToHost),
      "Failed to copy data from device to host.");

  return storage;
#endif
}

} // namespace spdl::core
