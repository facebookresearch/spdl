/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <limits>

#include "numpy_support.h"
#include "register_tar.h"
#include "zip_impl.h"

#include "memoryview_utils.h"

namespace nb = nanobind;

namespace spdl::archive {

namespace {

std::vector<size_t> get_fortran_strides(const NPYArray& array) {
  std::vector<size_t> strides;
  strides.reserve(array.shape.size());
  size_t stride = array.item_size;
  for (const size_t dim : array.shape) {
    strides.push_back(stride);
    if (dim != 0 && stride > std::numeric_limits<size_t>::max() / dim) {
      throw nb::value_error("NPY array strides exceed the supported range.");
    }
    stride *= dim;
  }
  return strides;
}

nb::dict _cast(const NPYArray& a) {
  nb::dict ret;
  ret["version"] = 3;
  ret["shape"] = nb::tuple(nb::cast(a.shape));
  ret["typestr"] = a.descr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)a.data, false};
  if (a.fortran_order) {
    ret["strides"] = nb::tuple(nb::cast(get_fortran_strides(a)));
  } else {
    ret["strides"] = nb::none();
  }
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", a.descr}};
  return ret;
}

NB_MODULE(_archive, m) {
  m.def("parse_zip", [](const nb::memoryview& data) {
    auto sv = ::spdl::detail::memoryview_to_sv(data);
    nb::gil_scoped_release _;
    return zip::parse_zip(sv.data(), sv.size());
  });

  nb::class_<NPYArray>(m, "NPYArray")
      .def_prop_ro("__array_interface__", [](NPYArray& self) -> nb::dict {
        return _cast(self);
      });

  m.def(
      "load_npy",
      [](uintptr_t p, size_t s, size_t o, std::optional<uint32_t> crc32) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* data = reinterpret_cast<const char*>(p) + o;
        if (crc32) {
          zip::verify_crc32(data, s, *crc32);
        }
        return load_npy(data, s);
      },
      nb::arg("data"),
      nb::arg("size"),
      nb::arg("offset") = 0,
      nb::arg("crc32") = nb::none(),
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "load_npy_compressed",
      [](uintptr_t p,
         size_t o,
         size_t cs,
         size_t ucs,
         std::optional<uint32_t> crc32) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* data = reinterpret_cast<const char*>(p) + o;
        return load_npy_compressed(data, cs, ucs, crc32);
      },
      nb::arg("data"),
      nb::arg("offset"),
      nb::arg("compressed_size"),
      nb::arg("uncompressed_size"),
      nb::arg("crc32") = nb::none(),
      nb::call_guard<nb::gil_scoped_release>());

  register_tar(m);
}

} // namespace
} // namespace spdl::archive
