/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/types.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace spdl::cuda {
using namespace spdl::core;
void register_types(nb::module_& m) {
  nb::class_<CUDAConfig>(m, "CUDAConfig")
      .def(
          nb::init<int, uintptr_t, std::optional<cuda_allocator>>(),
          nb::arg("device_index"),
          nb::arg("stream") = 0,
          nb::arg("allocator") = nb::none());
}
} // namespace spdl::cuda
