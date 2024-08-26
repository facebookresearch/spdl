/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/buffer.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// CUDABuffer
////////////////////////////////////////////////////////////////////////////////
CUDABuffer::CUDABuffer(
    std::vector<size_t> shape_,
    ElemClass elem_class_,
    size_t depth_,
    CUDAStorage* storage_,
    int device_index_)
    : Buffer(std::move(shape_), elem_class_, depth_, (Storage*)storage_),
      device_index(device_index_) {}

uintptr_t CUDABuffer::get_cuda_stream() const {
  return (uintptr_t)(((CUDAStorage*)(storage.get()))->stream);
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
    ElemClass elem_class,
    size_t depth) {
  return std::make_unique<CUDABuffer>(
      shape,
      elem_class,
      depth,
      new CUDAStorage{depth * prod(shape), cfg},
      cfg.device_index);
}

} // namespace spdl::core
