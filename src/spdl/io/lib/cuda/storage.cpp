/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/storage.h>
#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/storage.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::cuda {
void register_storage(nb::module_& m) {
  m.def(
      "cpu_storage",
      [](size_t size) -> std::shared_ptr<core::CPUStorage> {
#ifndef SPDL_USE_CUDA
        throw std::runtime_error("SPDL is not built with CUDA support.");
#else
        return std::make_shared<spdl::core::CPUStorage>(
            size, &alloc_pinned, &dealloc_pinned, true);
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("size"));
}

} // namespace spdl::cuda
