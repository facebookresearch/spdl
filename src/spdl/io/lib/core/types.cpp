/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/types.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;

namespace spdl::core {
void register_types(nb::module_& m) {
  nb::class_<DemuxConfig>(m, "DemuxConfig")
      .def(
          nb::init<
              const std::optional<std::string>,
              const std::optional<OptionDict>,
              int>(),
          nb::arg("format") = nb::none(),
          nb::arg("format_options") = nb::none(),
          nb::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE);

  nb::class_<DecodeConfig>(m, "DecodeConfig")
      .def(
          nb::init<
              const std::optional<std::string>&,
              const std::optional<OptionDict>&>(),
          nb::arg("decoder") = nb::none(),
          nb::arg("decoder_options") = nb::none());

  nb::exception<spdl::core::InternalError>(
      m, "InternalError", PyExc_AssertionError);
}
} // namespace spdl::core
