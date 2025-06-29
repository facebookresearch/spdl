/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

#include <libspdl/core/encoder.h>
#include <libspdl/core/muxer.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

namespace spdl::core {
void register_encoding(nb::module_& m) {
  nb::class_<Muxer>(m, "Muxer")
      .def(
          "open",
          &Muxer::open,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("muxer_config") = std::nullopt)
      .def(
          "add_encode_stream",
          &Muxer::add_encode_stream<MediaType::Audio>,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("config"),
          nb::arg("encoder") = std::nullopt,
          nb::arg("encoder_config") = std::nullopt)
      .def(
          "add_encode_stream",
          &Muxer::add_encode_stream<MediaType::Video>,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("config"),
          nb::arg("encoder") = std::nullopt,
          nb::arg("encoder_config") = std::nullopt)
      .def(
          "add_remux_stream",
          &Muxer::add_remux_stream<MediaType::Audio>,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("codec"))
      .def(
          "add_remux_stream",
          &Muxer::add_remux_stream<MediaType::Video>,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("codec"))
      .def(
          "write",
          &Muxer::write<MediaType::Audio>,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "write",
          &Muxer::write<MediaType::Video>,
          nb::call_guard<nb::gil_scoped_release>())
      .def("flush", &Muxer::flush, nb::call_guard<nb::gil_scoped_release>())
      .def("close", &Muxer::close, nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "muxer",
      [](const std::string& uri, const std::optional<std::string>& format) {
        return std::make_unique<Muxer>(uri, format);
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg(), // Positional-only
      nb::kw_only(),
      nb::arg("format") = std::nullopt);

  nb::class_<AudioEncodeConfig>(m, "AudioEncodeConfig");

  m.def(
      "audio_encode_config",
      [](int num_channels,
         const std::optional<std::string>& sample_fmt,
         const std::optional<int> sample_rate,
         int bit_rate,
         int compression_level,
         int qscale) {
        return AudioEncodeConfig{
            .num_channels = num_channels,
            .sample_fmt = sample_fmt,
            .sample_rate = sample_rate,
            .bit_rate = bit_rate,
            .compression_level = compression_level,
            .qscale = qscale};
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::kw_only(),
      nb::arg("num_channels"),
      nb::arg("sample_fmt") = std::nullopt,
      nb::arg("sample_rate") = std::nullopt,
      nb::arg("bit_rate") = -1,
      nb::arg("compression_level") = -1,
      nb::arg("qscale") = -1);

  nb::class_<VideoEncodeConfig>(m, "VideoEncodeConfig");

  m.def(
      "video_encode_config",
      [](int height,
         int width,
         const std::optional<std::tuple<int, int>> frame_rate,
         const std::optional<std::string>& pix_fmt,
         int bit_rate,
         int compression_level,
         int qscale,
         int gop_size,
         int max_b_frames,
         const std::optional<std::string>& colorspace,
         const std::optional<std::string>& color_primaries,
         const std::optional<std::string>& color_trc) {
        auto fr = [&]() -> std::optional<Rational> {
          if (frame_rate) {
            auto rate = *frame_rate;
            return Rational{std::get<0>(rate), std::get<1>(rate)};
          }
          return std::nullopt;
        }();
        return VideoEncodeConfig{
            .height = height,
            .width = width,
            .pix_fmt = pix_fmt,
            .frame_rate = fr,
            .bit_rate = bit_rate,
            .compression_level = compression_level,
            .qscale = qscale,
            .gop_size = gop_size,
            .max_b_frames = max_b_frames,
            .colorspace = colorspace,
            .color_primaries = color_primaries,
            .color_trc = color_trc};
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::kw_only(),
      nb::arg("height"),
      nb::arg("width"),
      nb::arg("frame_rate") = std::nullopt,
      nb::arg("pix_fmt") = std::nullopt,
      nb::arg("bit_rate") = -1,
      nb::arg("compression_level") = -1,
      nb::arg("qscale") = -1,
      nb::arg("gop_size") = -1,
      nb::arg("max_b_frames") = -1,
      nb::arg("colorspace") = std::nullopt,
      nb::arg("color_primaries") = std::nullopt,
      nb::arg("color_trc") = std::nullopt);

  nb::class_<VideoEncoder>(m, "VideoEncoder")
      .def(
          "encode",
          &VideoEncoder::encode,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "flush",
          &VideoEncoder::flush,
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<AudioEncoder>(m, "AudioEncoder")
      .def(
          "encode",
          &AudioEncoder::encode,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "flush",
          &AudioEncoder::flush,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "frame_size",
          &AudioEncoder::get_frame_size,
          nb::call_guard<nb::gil_scoped_release>());
}

} // namespace spdl::core
