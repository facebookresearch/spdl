/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/filter_graph.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
void register_filtering(nb::module_& m) {
  nb::class_<FilterGraph>(m, "FiilterGraph")
      .def(
          "add_frames",
          &FilterGraph::add_frames,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("frames"),
          nb::kw_only(),
          nb::arg("key") = std::nullopt)
      .def(
          "flush",
          &FilterGraph::flush,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "get_frames",
          &FilterGraph::get_frames,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::kw_only(),
          nb::arg("key") = std::nullopt)
      .def(
          "dump", &FilterGraph::dump, nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "make_filter_graph",
      &make_filter_graph,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("filter_desc"));
}
} // namespace spdl::core
