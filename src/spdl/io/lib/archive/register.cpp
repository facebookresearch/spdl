/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "numpy_support.h"
#include "zip_impl.h"

namespace nb = nanobind;

namespace spdl::archive {

namespace {

nb::dict _cast(const NPYArray& a) {
  nb::dict ret;
  ret["version"] = 3;
  ret["shape"] = nb::tuple(nb::cast(a.shape));
  ret["typestr"] = a.descr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)a.data, false};
  ret["strides"] = nb::none();
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", a.descr}};
  return ret;
}

NB_MODULE(_archive, m) {
  m.def(
      "parse_zip",
      [](const nb::bytes& bytes) {
        return zip::parse_zip(bytes.c_str(), bytes.size());
      },
      nb::call_guard<nb::gil_scoped_release>());

  nb::class_<NPYArray>(m, "NPYArray")
      .def_prop_ro("__array_interface__", [](NPYArray& self) -> nb::dict {
        return _cast(self);
      });

  m.def(
      "load_npy",
      [](uintptr_t p, size_t s, size_t o) {
        return load_npy((const char*)p + o, s);
      },
      nb::arg("data"),
      nb::arg("size"),
      nb::arg("offset") = 0,
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "load_npy_compressed",
      [](uintptr_t p, size_t o, uint32_t cs, uint32_t ucs) {
        return load_npy_compressed((const char*)p + o, cs, ucs);
      },
      nb::arg("data"),
      nb::arg("offset"),
      nb::arg("compressed_size"),
      nb::arg("uncompressed_size"),
      nb::call_guard<nb::gil_scoped_release>());
}

} // namespace
} // namespace spdl::archive
