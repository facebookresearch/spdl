/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

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
    if constexpr (media == MediaType::Video) {
      const auto& r = c.get_frame_rate();
      p.emplace_back(fmt::format("frame_rate={}/{}", r.num, r.den));
    }
    p.emplace_back(fmt::format("width={}", c.get_width()));
    p.emplace_back(fmt::format("height={}", c.get_height()));
  }
  return fmt::format("Codec=<{}>", fmt::join(p, ", "));
}

std::string get_ts(const std::tuple<double, double>& ts) {
  return fmt::format("timestamp=({}, {})", std::get<0>(ts), std::get<1>(ts));
}

template <MediaType media>
const Codec<media>& get_ref(const std::optional<Codec<media>>& c) {
  if (!c) {
    throw std::runtime_error("Packet does not have codec info.");
  }
  return *c;
}

template <MediaType media>
size_t num_packets(const Packets<media>& packets) {
  return get_timestamps(packets).size();
}

} // namespace

void register_packets(nb::module_& m) {
  nb::class_<AudioPackets>(m, "AudioPackets")
      .def(
          "__repr__",
          [](const AudioPackets& self) {
            std::vector<std::string> parts{
                fmt::format("src=\"{}:{}\"", self.src, self.stream_index)};
            if (auto ts = get_timestamps(self); !ts.empty()) {
              parts.push_back(fmt::format("num_packets={}", num_packets(self)));
              parts.push_back(fmt::format(
                  "pts=[{:.3f}, {:.3f}~]", ts[0], ts[ts.size() - 1]));
              // Note: Audio end time is not precise due to the fact that
              // one packet contains multiple samples.
              // So we add tilde
            }
            if (self.timestamp) {
              parts.push_back(get_ts(*self.timestamp));
            }
            if (self.codec) {
              parts.push_back(get_summary(*self.codec));
            }
            return fmt::format("AudioPackets<{}>", fmt::join(parts, ", "));
          })
      .def(
          "__len__", [](const AudioPackets& self) { return num_packets(self); })
      .def_prop_ro(
          "timestamp",
          [](const AudioPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "sample_rate",
          [](const AudioPackets& self) {
            return get_ref(self.codec).get_sample_rate();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "num_channels",
          [](const AudioPackets& self) {
            return get_ref(self.codec).get_num_channels();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const AudioPackets& self) -> std::optional<AudioCodec> {
            if (self.codec) {
              return AudioCodec{*self.codec};
            }
            return std::nullopt;
          },
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
            return get_timestamps(self);
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "get_timestamps",
          [](const VideoPackets& self, bool raw) -> std::vector<double> {
            return get_timestamps(self, raw);
          },
          nb::call_guard<nb::gil_scoped_release>(),
          nb::kw_only(),
          nb::arg("raw") = false)
      // Note: this is a window timestamp specified by user.
      .def_prop_ro(
          "timestamp",
          [](const VideoPackets& self) {
            nb::gil_scoped_release __g;
            return self.timestamp;
          })
      .def_prop_ro(
          "pix_fmt",
          [](const VideoPackets& self) {
            return get_ref(self.codec).get_pix_fmt();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          [](const VideoPackets& self) {
            return get_ref(self.codec).get_width();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          [](const VideoPackets& self) {
            return get_ref(self.codec).get_height();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "frame_rate",
          [](const VideoPackets& self) {
            auto rate = get_ref(self.codec).get_frame_rate();
            return std::tuple<int, int>(rate.num, rate.den);
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const VideoPackets& self) -> std::optional<VideoCodec> {
            if (self.codec) {
              return VideoCodec{*self.codec};
            }
            return std::nullopt;
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__len__", [](const VideoPackets& self) { return num_packets(self); })
      .def(
          "__repr__",
          [](const VideoPackets& self) {
            std::vector<std::string> parts{
                fmt::format("src=\"{}:{}\"", self.src, self.stream_index)};
            if (auto ts = get_timestamps(self); !ts.empty()) {
              parts.push_back(fmt::format("num_packets={}", num_packets(self)));
              parts.push_back(fmt::format(
                  "pts=[{:.3f}, {:.3f}]", ts[0], ts[ts.size() - 1]));
            }
            if (self.timestamp) {
              parts.push_back(get_ts(*self.timestamp));
            }
            if (self.codec) {
              parts.push_back(get_summary(*self.codec));
            }
            return fmt::format("VideoPackets<{}>", fmt::join(parts, ", "));
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
          [](const ImagePackets& self) { return get_timestamps(self).at(0); },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "pix_fmt",
          [](const ImagePackets& self) {
            return get_ref(self.codec).get_pix_fmt();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          [](const ImagePackets& self) {
            return get_ref(self.codec).get_width();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          [](const ImagePackets& self) {
            return get_ref(self.codec).get_height();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "codec",
          [](const ImagePackets& self) -> std::optional<ImageCodec> {
            if (self.codec) {
              return ImageCodec{*self.codec};
            }
            return std::nullopt;
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__repr__",
          [](const ImagePackets& self) {
            return fmt::format(
                "ImagePackets<src=\"{}\", {}>",
                self.src,
                get_summary(*self.codec));
          })
      .def(
          "clone",
          [](const ImagePackets& self) -> ImagePacketsPtr {
            return std::make_unique<ImagePackets>(self);
          },
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::core
