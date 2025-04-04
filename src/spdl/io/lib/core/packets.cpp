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

namespace {
template <MediaType media>
std::string get_summary(const Codec<media>& c) {
  std::vector<std::string> p;

  p.emplace_back(fmt::format("bit_rate={}", c.get_bit_rate()));
  p.emplace_back(fmt::format("bits_per_sample={}", c.get_bits_per_sample()));
  p.emplace_back(fmt::format("codec=\"{}\"", c.get_name()));

  if constexpr (media == MediaType::Audio) {
    p.emplace_back(fmt::format("sample_format=\"{}\"", c.get_sample_fmt()));
    p.emplace_back(fmt::format("sample_rate={}", c.get_sample_rate()));
    p.emplace_back(fmt::format("num_channels={}", c.get_num_channels()));
  }
  if constexpr (media == MediaType::Video || media == MediaType::Image) {
    p.emplace_back(fmt::format("pixel_format=\"{}\"", c.get_pix_fmt()));
    p.emplace_back(fmt::format("width={}", c.get_width()));
    p.emplace_back(fmt::format("height={}", c.get_height()));
  }
  return fmt::format("Codec=<{}>", fmt::join(p, ", "));
}

std::string get_ts(const std::optional<std::tuple<double, double>>& ts) {
  return ts ? fmt::format("({}, {})", std::get<0>(*ts), std::get<1>(*ts))
            : "n/a";
}

size_t num_packets(const VideoPackets& packets) {
  if (!packets.timestamp) {
    return packets.get_packets().size();
  }
  size_t ret = 0;
  auto [start, end] = *packets.timestamp;
  auto tb = packets.time_base;
  auto pkts = packets.iter_data();
  while (pkts) {
    auto pts = static_cast<double>(pkts().pts) * tb.num / tb.den;
    if (start <= pts && pts < end) {
      ++ret;
    }
  }
  return ret;
}

template <MediaType media>
std::vector<double> get_pts(const Packets<media>& packets) {
  std::vector<double> ret;
  auto base = packets.codec.get_time_base();
  auto pkts = packets.iter_data();
  while (pkts) {
    ret.push_back(double(pkts().pts) * base.num / base.den);
  }
  return ret;
}

} // namespace

void register_packets(nb::module_& m) {
  nb::class_<AudioPackets>(m, "AudioPackets")
      .def(
          "__repr__",
          [](const AudioPackets& self) {
            return fmt::format(
                "AudioPackets<src=\"{}\", timestamp={}, {}>",
                self.src,
                get_ts(self.timestamp),
                get_summary(self.codec));
          })
      .def_prop_ro(
          "timestamp",
          [](const AudioPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "sample_rate",
          [](const AudioPackets& self) { return self.codec.get_sample_rate(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "num_channels",
          [](const AudioPackets& self) {
            return self.codec.get_num_channels();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const AudioPackets& self) { return AudioCodec{self.codec}; },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "clone",
          [](const AudioPackets& self) -> AudioPacketsPtr {
            return std::make_unique<AudioPackets>(self);
          },
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<VideoPackets>(m, "VideoPackets")
      .def(
          "_get_pts",
          [](const VideoPackets& self) -> std::vector<double> {
            return get_pts(self);
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
          [](const VideoPackets& self) { return self.codec.get_pix_fmt(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          [](const VideoPackets& self) { return self.codec.get_width(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          [](const VideoPackets& self) { return self.codec.get_height(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "frame_rate",
          [](const VideoPackets& self) {
            auto rate = self.codec.get_frame_rate();
            return std::tuple<int, int>(rate.num, rate.den);
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const VideoPackets& self) { return VideoCodec{self.codec}; },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__len__", [](const VideoPackets& self) { return num_packets(self); })
      .def(
          "__repr__",
          [](const VideoPackets& self) {
            const auto& frame_rate = self.codec.get_frame_rate();
            return fmt::format(
                "VideoPackets<src=\"{}\", timestamp={}, frame_rate={}/{}, num_packets={}, {}>",
                self.src,
                get_ts(self.timestamp),
                frame_rate.num,
                frame_rate.den,
                num_packets(self),
                get_summary(self.codec));
          })
      .def(
          "clone",
          [](const VideoPackets& self) -> VideoPacketsPtr {
            return std::make_unique<VideoPackets>(self);
          },
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
          [](const ImagePackets& self) { return get_pts(self).at(0); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "pix_fmt",
          [](const ImagePackets& self) { return self.codec.get_pix_fmt(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          [](const ImagePackets& self) { return self.codec.get_width(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          [](const ImagePackets& self) { return self.codec.get_height(); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const ImagePackets& self) { return ImageCodec{self.codec}; },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__repr__",
          [](const ImagePackets& self) {
            return fmt::format(
                "ImagePackets<src=\"{}\", {}>",
                self.src,
                get_summary(self.codec));
          })
      .def(
          "clone",
          [](const ImagePackets& self) -> ImagePacketsPtr {
            return std::make_unique<ImagePackets>(self);
          },
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::core
