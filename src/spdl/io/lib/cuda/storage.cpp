/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/core/storage.h>
#include <libspdl/cuda/storage.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

#ifdef SPDL_USE_CUDA
#define _V(var_name) var_name
#define _I(impl) impl
#else
#define _V(var_name)
#define _I(impl) \
  { throw std::runtime_error("SPDL is not built with CUDA support."); }
#endif

namespace spdl::cuda {
void register_storage(nb::module_& m) {
  m.def(
      "cpu_storage",
      [](size_t _V(size)) -> std::shared_ptr<core::CPUStorage> _I({
        return std::make_shared<spdl::core::CPUStorage>(
            size, &alloc_pinned, &dealloc_pinned, true);
      }),
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("size"));
}

} // namespace spdl::cuda
