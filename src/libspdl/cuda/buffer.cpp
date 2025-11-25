/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/buffer.h>

namespace spdl::cuda {

////////////////////////////////////////////////////////////////////////////////
// CUDABuffer
////////////////////////////////////////////////////////////////////////////////
void* CUDABuffer::data() {
  return storage->data();
}

void* CUDABuffer::data() const {
  return storage->data();
}

uintptr_t CUDABuffer::get_cuda_stream() const {
  return (uintptr_t)(((CUDAStorage*)(storage.get()))->stream_);
}

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////
namespace {
inline size_t prod(const std::vector<size_t>& shape) {
  size_t val = 1;
  for (auto& v : shape) {
    val *= v;
  }
  return val;
}
} // namespace

CUDABufferPtr cuda_buffer(
    const std::vector<size_t>& shape,
    const CUDAConfig& cfg,
    spdl::core::ElemClass elem_class,
    size_t depth) {
  return std::make_unique<CUDABuffer>(
      cfg.device_index,
      std::make_shared<CUDAStorage>(depth * prod(shape), cfg),
      shape,
      elem_class,
      depth);
}

} // namespace spdl::cuda
