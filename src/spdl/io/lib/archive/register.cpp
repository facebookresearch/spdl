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

#include "zip_impl.h"

namespace nb = nanobind;

namespace spdl::archive {

namespace {

NB_MODULE(_archive, m) {
  m.def("parse_zip", [](const nb::bytes& bytes) {
    auto* data = bytes.c_str();
    auto size = bytes.size();

    nb::gil_scoped_release __g;
    return zip::parse_zip(data, size);
  });

  m.def(
      "inflate",
      [](const nb::bytes& bytes,
         uint64_t offset,
         uint32_t compressed_size,
         uint32_t uncompressed_size) {
        auto* data = bytes.c_str();
        nb::bytearray ret{};
        ret.resize(uncompressed_size);

        {
          nb::gil_scoped_release __g;
          zip::inflate(
              data + offset, compressed_size, ret.data(), uncompressed_size);
        }
        return ret;
      });
}

} // namespace
} // namespace spdl::archive
