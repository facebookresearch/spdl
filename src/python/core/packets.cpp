#include <libspdl/core/packets.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <fmt/format.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
std::string get_codec_info(AVCodecParameters* codecpar) {
  if (!codecpar) {
    return "<No codec information>";
  }

  std::vector<std::string> parts;

  parts.emplace_back(fmt::format("bit_rate={}", codecpar->bit_rate));
  parts.emplace_back(
      fmt::format("bits_per_sample={}", codecpar->bits_per_raw_sample));
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  parts.emplace_back(
      fmt::format("codec=\"{}\"", desc ? desc->name : "unknown"));

  if constexpr (media_type == MediaType::Audio) {
    parts.emplace_back(fmt::format("sample_rate={}", codecpar->sample_rate));
    parts.emplace_back(fmt::format(
        "num_channels={}",
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 2, 100)
        codecpar->ch_layout.nb_channels
#else
        codecpar->channels
#endif
        ));
  }
  if constexpr (
      media_type == MediaType::Video || media_type == MediaType::Image) {
    parts.emplace_back(
        fmt::format("width={}, height={}", codecpar->width, codecpar->height));
  }
  return fmt::format("{}", fmt::join(parts, ", "));
}

} // namespace

void register_packets(nb::module_& m) {
  nb::class_<AudioPackets>(m, "AudioPackets")
      .def(
          "__repr__",
          [](const AudioPackets& self) {
            return fmt::format(
                "AudioPackets<src=\"{}\", timestamp=({}, {}), sample_format=\"{}\", {}>",
                self.src,
                std::get<0>(self.timestamp),
                std::get<1>(self.timestamp),
                self.get_media_format_name(),
                get_codec_info<MediaType::Audio>(self.codecpar));
          })
      .def_prop_ro(
          "timestamp", [](const AudioPackets& self) { return self.timestamp; })
      .def("clone", &clone<MediaType::Audio>);

  nb::class_<VideoPackets>(m, "VideoPackets")
      .def(
          "_get_pts",
          [](const VideoPackets& self) -> std::vector<double> {
            std::vector<double> ret;
            auto base = self.time_base;
            for (auto& packet : self.get_packets()) {
              ret.push_back(double(packet->pts) * base.num / base.den);
            }
            return ret;
          })
      .def_prop_ro(
          "timestamp", [](const VideoPackets& self) { return self.timestamp; })
      .def(
          "__len__",
          [](const VideoPackets& self) { return self.num_packets(); })
      .def(
          "__repr__",
          [](const VideoPackets& self) {
            return fmt::format(
                "VideoPackets<src=\"{}\", timestamp=({}, {}), frame_rate={}/{}, num_packets={}, pixel_format=\"{}\", {}>",
                self.src,
                std::get<0>(self.timestamp),
                std::get<1>(self.timestamp),
                self.frame_rate.num,
                self.frame_rate.den,
                self.num_packets(),
                self.get_media_format_name(),
                get_codec_info<MediaType::Video>(self.codecpar));
          })
      .def("clone", [](const VideoPackets& self) { return clone(self); })
      .def("_split_at_keyframes", [](const VideoPackets& self) {
        return spdl::core::split_at_keyframes(self);
      });

  nb::class_<ImagePackets>(m, "ImagePackets")
      .def(
          "_get_pts",
          [](const ImagePackets& self) {
            auto base = self.time_base;
            auto pts = self.get_packets().at(0)->pts;
            return pts * base.num / base.den;
          })
      .def(
          "__repr__",
          [](const ImagePackets& self) {
            return fmt::format(
                "ImagePackets<src=\"{}\", pixel_format=\"{}\", {}>",
                self.src,
                self.get_media_format_name(),
                get_codec_info<MediaType::Image>(self.codecpar));
          })
      .def("clone", [](const ImagePackets& self) { return clone(self); });
}
} // namespace spdl::core
