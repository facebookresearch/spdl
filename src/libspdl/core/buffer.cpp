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
// CPUBuffer
////////////////////////////////////////////////////////////////////////////////

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

CPUBufferPtr cpu_buffer(
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
      SPDL_FAIL(
          fmt::format(
              "The provided storage does not have enough capacity. ({} < {})",
              storage->size,
              size));
    }
  }

  // The following does not compile on Apple clang 15
  // return std::make_unique<CPUBuffer>(storage, shape, elem_class, depth);
  auto ret = std::make_unique<CPUBuffer>();
  ret->storage = storage ? storage : std::make_shared<CPUStorage>(size);
  ret->shape = shape;
  ret->elem_class = elem_class;
  ret->depth = depth;
  return ret;
}

} // namespace spdl::core
