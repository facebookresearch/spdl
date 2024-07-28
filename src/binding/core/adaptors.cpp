/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/adaptor.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::core {

void register_adaptors(nb::module_& m) {
  nb::class_<SourceAdaptor>(m, "SourceAdaptor").def(nb::init<>());

  nb::class_<MMapAdaptor>(m, "MMapAdaptor").def(nb::init<>());

  nb::class_<BytesAdaptor>(m, "BytesAdaptor").def(nb::init<>());
}
} // namespace spdl::core
