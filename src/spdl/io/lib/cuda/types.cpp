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

namespace nb = nanobind;

namespace spdl::cuda {
void register_types(nb::module_& m) {
  nb::class_<CUDAConfig>(m, "CUDAConfig");

  m.def(
      "cuda_config",
      [](int index,
         uintptr_t stream,
         std::optional<cuda_allocator> allocator) -> CUDAConfig {
#ifdef SPDL_USE_CUDA
        return CUDAConfig{index, stream, std::move(allocator)};
#else
        throw std::runtime_error("SPDL is not built with CUDA support.");
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("device_index"),
      nb::arg("stream") = 0,
      nb::arg("allocator") = nb::none());
}
} // namespace spdl::cuda
