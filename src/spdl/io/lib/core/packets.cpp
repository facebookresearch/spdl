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
          [](const AudioPackets& self) {
            nb::gil_scoped_release __g;
            return self.get_sample_rate();
          })
      .def_prop_ro(
          "num_channels",
          [](const AudioPackets& self) {
            nb::gil_scoped_release __g;
            return self.get_num_channels();
          })
      .def("clone", [](const AudioPackets& self) {
        nb::gil_scoped_release __g;
        return clone(self);
      });

  nb::class_<VideoPackets>(m, "VideoPackets")
      .def(
          "_get_pts",
          [](const VideoPackets& self) -> std::vector<double> {
            nb::gil_scoped_release __g;
            std::vector<double> ret;
            auto base = self.time_base;
            auto pkts = self.iter_packets();
            while (pkts) {
              ret.push_back(double(pkts().pts) * base.num / base.den);
            }
            return ret;
          })
      .def_prop_ro(
          "timestamp",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "pix_fmt",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          })
      .def_prop_ro(
          "width",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          })
      .def_prop_ro(
          "frame_rate",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            auto rate = self.get_frame_rate();
            return std::tuple<int, int>(rate.num, rate.den);
          })
      .def("__len__", &VideoPackets::num_packets)
      .def("__repr__", &VideoPackets::get_summary)
      .def("clone", [](const VideoPackets& self) {
        nb::gil_scoped_release __g;
        return clone(self);
      });

  m.def(
      "_extract_packets_at_indices",
      [](VideoPacketsPtr packets, const std::vector<size_t>& indices) {
        nb::gil_scoped_release __g;
        return extract_packets_at_indices(packets, indices);
      });

  nb::class_<ImagePackets>(m, "ImagePackets")
      .def(
          "_get_pts",
          [](const ImagePackets& self) {
            nb::gil_scoped_release __g;
            auto base = self.time_base;
            return double(self.get_pts()) * base.num / base.den;
          })
      .def_prop_ro(
          "pix_fmt",
          [](const ImagePackets& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          })
      .def_prop_ro(
          "width",
          [](const ImagePackets& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](const ImagePackets& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          })
      .def("__repr__", &ImagePackets::get_summary)
      .def("clone", [](const ImagePackets& self) {
        nb::gil_scoped_release __g;
        return clone(self);
      });
}
} // namespace spdl::core
