/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/cuda/types.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;

// Workaround for -Werror,-Wunused-variable in case SPDL_USE_CUDA
// is not defined. It hides the variable name.
#ifdef SPDL_USE_CUDA
#define _(var_name) var_name
#else
#define _(var_name)
#endif

namespace spdl::cuda {
void register_types(nb::module_& m) {
  nb::class_<CUDAConfig>(
      m,
      "CUDAConfig",
      "Specify the CUDA device and memory management.\n\n"
      "See the factory function :py:func:`~spdl.io.cuda_config`.");

  m.def(
      "cuda_config",
      [](int _(index),
         uintptr_t _(stream),
         std::optional<cuda_allocator> _(allocator)) -> CUDAConfig {
#ifdef SPDL_USE_CUDA
        return CUDAConfig{index, stream, std::move(allocator)};
#else
        throw std::runtime_error("SPDL is not built with CUDA support.");
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("device_index"),
      nb::arg("stream") = 0x2,
      nb::arg("allocator") = nb::none());
}
} // namespace spdl::cuda
