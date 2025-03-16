/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/storage.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace spdl::core {
void register_storage(nb::module_& m) {
  nb::class_<CPUStorage>(m, "CPUStorage");

  m.def(
      "cpu_storage",
      [](size_t size, bool pin_memory) {
        return pin_memory ? std::make_shared<CPUStorage>(
                                size, &alloc_pinned, &dealloc_pinned, true)
                          : std::make_shared<CPUStorage>(size);
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("size"),
      nb::arg("pin_memory") = false);
}

} // namespace spdl::core
