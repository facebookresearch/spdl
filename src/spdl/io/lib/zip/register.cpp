/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/core.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <glog/logging.h>

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

namespace {

NB_MODULE(_zip, m) {
  m.def("parse_zip", [](const nb::bytes& bytes) {
    auto* data = bytes.c_str();
    auto size = bytes.size();

    nb::gil_scoped_release __g;
    return parse_zip(data, size);
  });
}

} // namespace
} // namespace spdl::zip
