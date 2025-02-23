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

#include "spdl_gil.h"

namespace nb = nanobind;

namespace spdl::core {
namespace {
std::shared_ptr<CPUStorage> _cpu_storage(size_t size, bool pin_memory) {
  RELEASE_GIL();
  return std::make_shared<CPUStorage>(size, pin_memory);
}
} // namespace

void register_storage(nb::module_& m) {
  nb::class_<CPUStorage>(m, "CPUStorage");

  m.def(
      "cpu_storage",
      &_cpu_storage,
      nb::arg("size"),
      nb::arg("pin_memory") = false);
}

} // namespace spdl::core
