/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

namespace spdl::zip {

using ZipMetaData = std::tuple<
    uint64_t, // offset
    uint64_t, // compressed_size
    uint64_t, // uncompressed_size
    uint16_t // compression_method
    >;

std::map<std::string, ZipMetaData> parse_zip(
    const char* root,
    const size_t len);

void inflate(
    const char* root,
    uint32_t compressed_size,
    void* dst,
    uint32_t uncompressed_size);

namespace {

NB_MODULE(_zip, m) {
  m.def("parse_zip", [](const nb::bytes& bytes) {
    auto* data = bytes.c_str();
    auto size = bytes.size();

    nb::gil_scoped_release __g;
    return parse_zip(data, size);
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
          inflate(
              data + offset, compressed_size, ret.data(), uncompressed_size);
        }
        return ret;
      });
}

} // namespace
} // namespace spdl::zip
