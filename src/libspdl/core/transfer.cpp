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

namespace spdl::core {
namespace {
size_t prod(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto& v : shape) {
    ret *= v;
  }
  return ret;
}

} // namespace

CUDABufferPtr transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg) {
  return transfer_buffer(
      buffer->shape, buffer->elem_class, buffer->depth, buffer->data(), cfg);
}

CUDABufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  TRACE_EVENT("decoding", "core::transfer_buffer");

  auto ret = cuda_buffer(shape, cfg, elem_class, depth);

  size_t size = depth * prod(shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), ptr, size, cudaMemcpyHostToDevice),
      "Failed to copy data from host to device.");

  return ret;
#endif
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
