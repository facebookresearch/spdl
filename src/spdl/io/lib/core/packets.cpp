/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <fmt/format.h>

namespace nb = nanobind;

namespace spdl::core {
void register_packets(nb::module_& m) {
  nb::class_<AudioPackets>(m, "AudioPackets")
      .def("__repr__", &AudioPackets::get_summary)
      .def_prop_ro(
          "timestamp",
          [](const AudioPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "sample_rate",
          &AudioPackets::get_sample_rate,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "num_channels",
          &AudioPackets::get_num_channels,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          &AudioPackets::get_codec,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "clone",
          &AudioPackets::clone,
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<VideoPackets>(m, "VideoPackets")
      .def(
          "_get_pts",
          [](const VideoPackets& self) -> std::vector<double> {
            std::vector<double> ret;
            auto base = self.codec.get_time_base();
            auto pkts = self.iter_packets();
            while (pkts) {
              ret.push_back(double(pkts().pts) * base.num / base.den);
            }
            return ret;
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "timestamp",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "pix_fmt",
          &VideoPackets::get_media_format_name,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          &VideoPackets::get_width,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          &VideoPackets::get_height,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "frame_rate",
          [](const VideoPackets& self) {
            auto rate = self.get_frame_rate();
            return std::tuple<int, int>(rate.num, rate.den);
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          &VideoPackets::get_codec,
          nb::call_guard<nb::gil_scoped_release>())
      .def("__len__", &VideoPackets::num_packets)
      .def("__repr__", &VideoPackets::get_summary)
      .def(
          "clone",
          &VideoPackets::clone,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_extract_packets_at_indices",
      &extract_packets_at_indices,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
      nb::arg("indices"));

  nb::class_<ImagePackets>(m, "ImagePackets")
      .def(
          "_get_pts",
          [](const ImagePackets& self) {
            auto base = self.codec.get_time_base();
            return double(self.get_pts()) * base.num / base.den;
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "pix_fmt",
          &ImagePackets::get_media_format_name,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          &ImagePackets::get_width,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          &ImagePackets::get_height,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          &ImagePackets::get_codec,
          nb::call_guard<nb::gil_scoped_release>())
      .def("__repr__", &ImagePackets::get_summary)
      .def(
          "clone",
          &ImagePackets::clone,
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::core
