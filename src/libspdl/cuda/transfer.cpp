/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/transfer.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/detail/utils.h"

#include <glog/logging.h>

#include <mutex>

namespace spdl::cuda {
namespace {
size_t prod(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto& v : shape) {
    ret *= v;
  }
  return ret;
}

std::once_flag WARN_LEGACY_DEFAULT_STREAM_FLAG;
void warn_legacy_default_stream() noexcept {
  LOG(WARNING)
      << "The CPUStorage is page-locked (pinned), but the legacy default CUDA stream (0) is used. "
         "This is likely not what you intend. "
         "Please use a non-default CUDA stream to overlap the data transfer with kernel execution.";
}

CUDABufferPtr transfer_buffer_impl(
    const std::vector<size_t>& shape,
    spdl::core::ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg,
    bool pinned_memory = false) {
  TRACE_EVENT("decoding", "core::transfer_buffer");
  auto ret = cuda_buffer(shape, cfg, elem_class, depth);
  size_t size = depth * prod(shape);

  if (pinned_memory) {
    if (cfg.stream == 0) {
      std::call_once(
          WARN_LEGACY_DEFAULT_STREAM_FLAG, warn_legacy_default_stream);
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
}

} // namespace

CUDABufferPtr transfer_buffer(
    spdl::core::CPUBufferPtr buffer,
    const CUDAConfig& cfg) {
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
    spdl::core::ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg) {
  return transfer_buffer_impl(shape, elem_class, depth, ptr, cfg);
}

spdl::core::CPUBufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    spdl::core::ElemClass elem_class,
    size_t depth,
    const void* ptr) {
  TRACE_EVENT("decoding", "core::transfer_buffer");

  auto ret = cpu_buffer(shape, elem_class, depth);
  size_t size = depth * prod(shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), ptr, size, cudaMemcpyDeviceToHost),
      "Failed to copy data from device to host.");

  return ret;
}

spdl::core::CPUStorage cp_to_cpu(
    const void* src,
    const std::vector<size_t>& shape) {
  size_t size = prod(shape);
  spdl::core::CPUStorage storage{size};

  CHECK_CUDA(
      cudaMemcpy(storage.data(), src, size, cudaMemcpyDeviceToHost),
      "Failed to copy data from device to host.");

  return storage;
}

} // namespace spdl::cuda
