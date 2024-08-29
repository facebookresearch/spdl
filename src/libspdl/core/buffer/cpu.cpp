/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/buffer.h>

#include "libspdl/core/detail/logging.h"

#include <fmt/format.h>
#include <glog/logging.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(
    std::vector<size_t> shape_,
    ElemClass elem_class_,
    size_t depth_,
    Storage* storage_)
    : shape(std::move(shape_)),
      elem_class(elem_class_),
      depth(depth_),
      storage(storage_) {}

void* Buffer::data() {
  return storage->data();
}

////////////////////////////////////////////////////////////////////////////////
// CPUBuffer
////////////////////////////////////////////////////////////////////////////////
CPUBuffer::CPUBuffer(
    const std::vector<size_t>& shape_,
    ElemClass elem_class_,
    size_t depth_,
    CPUStorage* storage_)
    : Buffer(shape_, elem_class_, depth_, (Storage*)storage_) {}

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

std::unique_ptr<CPUBuffer> cpu_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    bool pin_memory) {
  size_t size = depth * prod(shape);
  VLOG(0) << fmt::format(
      "Allocating {} bytes. (shape: {}, elem: {})",
      size,
      fmt::join(shape, ", "),
      depth);
  return std::make_unique<CPUBuffer>(
      shape, elem_class, depth, new CPUStorage{size, pin_memory});
}

} // namespace spdl::core
