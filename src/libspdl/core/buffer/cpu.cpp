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
Buffer::Buffer(std::vector<size_t> shape_, ElemClass elem_class_, size_t depth_)
    : shape(std::move(shape_)), elem_class(elem_class_), depth(depth_) {}

////////////////////////////////////////////////////////////////////////////////
// CPUBuffer
////////////////////////////////////////////////////////////////////////////////
CPUBuffer::CPUBuffer(
    const std::vector<size_t>& shape_,
    ElemClass elem_class_,
    size_t depth_,
    std::shared_ptr<CPUStorage> storage_)
    : Buffer(shape_, elem_class_, depth_), storage(std::move(storage_)) {}

void* CPUBuffer::data() {
  return storage->data();
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

std::unique_ptr<CPUBuffer> cpu_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    std::shared_ptr<CPUStorage> storage) {
  size_t size = depth * prod(shape);
  VLOG(5) << fmt::format(
      "Allocating {} bytes. (shape: {}, elem: {})",
      size,
      fmt::join(shape, ", "),
      depth);

  if (storage) {
    if (storage->size < size) [[unlikely]] {
      SPDL_FAIL(fmt::format(
          "The provided storage does not have enough capacity. ({} < {})",
          storage->size,
          size));
    }
  }

  return std::make_unique<CPUBuffer>(
      shape,
      elem_class,
      depth,
      storage ? std::move(storage) : std::make_shared<CPUStorage>(size));
}

} // namespace spdl::core
