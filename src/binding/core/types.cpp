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
#include <nanobind/stl/pair.h>
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

  nb::class_<CUDAConfig>(m, "CUDAConfig")
      .def(
          nb::init<int, uintptr_t, std::optional<cuda_allocator>>(),
          nb::arg("device_index"),
          nb::arg("stream") = 0,
          nb::arg("allocator") = nb::none());

  nb::class_<EncodeConfig>(m, "EncodeConfig")
      .def(
          nb::init<
              std::optional<std::string>,
              std::optional<OptionDict>,
              std::optional<std::string>,
              std::optional<OptionDict>,
              std::optional<std::string>,
              int,
              int,
              std::optional<std::string>,
              std::optional<std::string>,
              int,
              int,
              int,
              int,
              int>(),
          nb::arg("muxer") = nb::none(),
          nb::arg("muxer_options") = nb::none(),
          nb::arg("encoder") = nb::none(),
          nb::arg("encoder_options") = nb::none(),
          nb::arg("format") = nb::none(),
          nb::arg("width") = -1,
          nb::arg("height") = -1,
          nb::arg("scale_algo") = nb::none(),
          nb::arg("filter_desc") = nb::none(),
          nb::arg("bit_rate") = -1,
          nb::arg("compression_level") = -1,
          nb::arg("qscale") = -1,
          nb::arg("gop_size") = -1,
          nb::arg("max_bframes") = -1);

  nb::exception<spdl::core::InternalError>(
      m, "InternalError", PyExc_AssertionError);
}
} // namespace spdl::core
