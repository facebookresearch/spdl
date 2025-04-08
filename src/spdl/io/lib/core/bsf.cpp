/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/bsf.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

namespace spdl::core {
void register_bsf(nb::module_& m) {
  ///////////////////////////////////////////////////////////////////////////////
  // Bit Stream Filtering
  ///////////////////////////////////////////////////////////////////////////////
  nb::class_<BSF<MediaType::Video>>(m, "VideoBSF")
      .def(
          "filter",
          &BSF<MediaType::Video>::filter,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          nb::kw_only(),
          nb::arg("flush") = false)
      .def(
          "flush",
          &BSF<MediaType::Video>::flush,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<BSF<MediaType::Audio>>(m, "AudioBSF")
      .def(
          "filter",
          &BSF<MediaType::Audio>::filter,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          nb::kw_only(),
          nb::arg("flush") = false)
      .def(
          "flush",
          &BSF<MediaType::Audio>::flush,
          nb::call_guard<nb::gil_scoped_release>());
  ;
  nb::class_<BSF<MediaType::Image>>(m, "ImageBSF")
      .def(
          "filter",
          &BSF<MediaType::Image>::filter,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          nb::kw_only(),
          nb::arg("flush") = false)
      .def(
          "flush",
          &BSF<MediaType::Image>::flush,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_make_bsf",
      [](const VideoCodec& codec, const std::string& name) {
        return std::make_unique<BSF<MediaType::Video>>(codec, name);
      },
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "_make_bsf",
      [](const AudioCodec& codec, const std::string& name) {
        return std::make_unique<BSF<MediaType::Audio>>(codec, name);
      },
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "_make_bsf",
      [](const ImageCodec& codec, const std::string& name) {
        return std::make_unique<BSF<MediaType::Image>>(codec, name);
      },
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "apply_bsf",
      [](VideoPacketsPtr packets, const std::string& name) {
        if (!packets->codec) {
          throw std::runtime_error("The packets do not have codec.");
        }
        auto bsf = BSF<MediaType::Video>{*packets->codec, name};
        return bsf.filter(std::move(packets), true);
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
      nb::arg("bsf"));
}
} // namespace spdl::core
