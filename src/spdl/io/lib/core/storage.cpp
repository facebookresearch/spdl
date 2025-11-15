/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

#include <libspdl/core/storage.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace spdl::core {
void register_storage(nb::module_& m) {
  nb::class_<CPUStorage>(m, "CPUStorage");

  m.def(
      "cpu_storage",
      [](size_t size) { return std::make_shared<CPUStorage>(size); },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("size"));
}

} // namespace spdl::core
