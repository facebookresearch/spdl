/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frames.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
void register_frames(nb::module_& m) {
  nb::class_<AudioFrames>(m, "AudioFrames")
      .def_prop_ro(
          "num_frames",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_frames();
          })
      .def_prop_ro(
          "sample_rate",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_sample_rate();
          })
      .def_prop_ro(
          "num_channels",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_channels();
          })
      .def_prop_ro(
          "sample_fmt",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          })
      .def("__len__", [](AudioFrames& self) { return self.get_num_frames(); })
      .def(
          "__repr__",
          [](const AudioFrames& self) -> std::string {
            auto num_frames = self.get_num_frames();
            auto pts = [&]() {
              if (num_frames == 0) {
                return std::numeric_limits<double>::quiet_NaN();
              }
              auto tb = self.get_time_base();
              return double(self.get_pts()) * tb.num / tb.den;
            }();
            return fmt::format(
                "AudioFrames<num_frames={}, sample_format=\"{}\", sample_rate={}, num_channels={}, timestamp={:.3f}>",
                num_frames,
                self.get_media_format_name(),
                self.get_sample_rate(),
                self.get_num_channels(),
                pts);
          })
      .def(
          "clone",
          &AudioFrames::clone,
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<VideoFrames>(m, "VideoFrames")
      .def_prop_ro(
          "num_frames",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_frames();
          })
      .def_prop_ro(
          "num_planes",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_planes();
          })
      .def_prop_ro(
          "width",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          })
      .def_prop_ro(
          "pix_fmt",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          })
      .def("__len__", [](VideoFrames& self) { return self.get_num_frames(); })
      .def(
          "__getitem__",
          [](const VideoFrames& self, const nb::slice& slice) {
            auto [start, stop, step, len] =
                slice.compute(self.get_num_frames());
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          })
      .def(
          "__getitem__",
          [](const VideoFrames& self, int64_t i) { return self.slice(i); })
      .def(
          "__getitem__",
          [](const VideoFrames& self, std::vector<int64_t> idx) {
            return self.slice(idx);
          })
      .def(
          "_get_pts",
          [](const VideoFrames& self) -> std::vector<double> {
            nb::gil_scoped_release __g;
            std::vector<double> ret;
            auto tb = self.get_time_base();
            for (size_t i = 0; i < self.get_num_frames(); ++i) {
              ret.push_back(double(self.get_pts(i)) * tb.num / tb.den);
            }
            return ret;
          })
      .def(
          "__repr__",
          [](const VideoFrames& self) -> std::string {
            auto num_frames = self.get_num_frames();
            auto pts = [&]() {
              if (num_frames == 0) {
                return std::numeric_limits<double>::quiet_NaN();
              }
              auto tb = self.get_time_base();
              return double(self.get_pts()) * tb.num / tb.den;
            }();
            return fmt::format(
                "VideoFrames<num_frames={}, pixel_format=\"{}\", num_planes={}, width={}, height={}, timestamp={:.3f}>",
                num_frames,
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height(),
                pts);
          })
      .def(
          "clone",
          &VideoFrames::clone,
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<ImageFrames>(m, "ImageFrames")
      .def_prop_ro(
          "num_planes",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_planes();
          })
      .def_prop_ro(
          "width",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          })
      .def_prop_ro(
          "pix_fmt",
          [](ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          })
      .def_prop_ro(
          "metadata",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_metadata();
          })
      .def(
          "__repr__",
          [](const ImageFrames& self) {
            auto tb = self.get_time_base();
            if (tb.num) {
              return fmt::format(
                  "ImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}, timestamp={:.3f}>",
                  self.get_media_format_name(),
                  self.get_num_planes(),
                  self.get_width(),
                  self.get_height(),
                  double(self.get_pts()) * tb.num / tb.den);
            }
            return fmt::format(
                "ImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}>",
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height());
          })
      .def(
          "clone",
          &ImageFrames::clone,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("pts", [](const ImageFrames& self) -> double {
        nb::gil_scoped_release __g;
        auto tb = self.get_time_base();
        return double(self.get_pts()) * tb.num / tb.den;
      });
}
} // namespace spdl::core
