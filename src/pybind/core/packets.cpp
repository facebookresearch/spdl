#include <libspdl/core/packets.h>

#include <pybind11/pybind11.h>

#include <fmt/format.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace py = pybind11;

namespace spdl::core {
namespace {
template <MediaType media_type>
std::string get_codec_info(AVCodecParameters* codecpar) {
  if (!codecpar) {
    return "<No codec information>";
  }

  std::vector<std::string> parts;

  parts.emplace_back(fmt::format("bit_rate: {}", codecpar->bit_rate));
  parts.emplace_back(
      fmt::format("bits_per_sample: {}", codecpar->bits_per_raw_sample));
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  parts.emplace_back(fmt::format("codec: {}", desc ? desc->name : "unknown"));

  if constexpr (media_type == MediaType::Audio) {
    parts.emplace_back(fmt::format("sample_rate: {}", codecpar->sample_rate));
    parts.emplace_back(fmt::format("num_channels: {}", codecpar->channels));
  }
  if constexpr (
      media_type == MediaType::Video || media_type == MediaType::Image) {
    parts.emplace_back(fmt::format(
        "width: {}, height: {}", codecpar->width, codecpar->height));
  }
  return fmt::format("{}", fmt::join(parts, ", "));
}

} // namespace

void register_packets(py::module& m) {
  using AudioPacketsWrapper = PacketsWrapper<MediaType::Audio>;
  using VideoPacketsWrapper = PacketsWrapper<MediaType::Video>;
  using ImagePacketsWrapper = PacketsWrapper<MediaType::Image>;

  auto _AudioPacketsWrapper =
      py::class_<AudioPacketsWrapper, AudioPacketsWrapperPtr>(
          m, "AudioPacketsWrapper", py::module_local());
  auto _VideoPacketsWrapper =
      py::class_<VideoPacketsWrapper, VideoPacketsWrapperPtr>(
          m, "VideoPacketsWrapper", py::module_local());
  auto _ImagePacketsWrapper =
      py::class_<ImagePacketsWrapper, ImagePacketsWrapperPtr>(
          m, "ImagePacketsWrapper", py::module_local());

  _AudioPacketsWrapper.def("__repr__", [](const AudioPacketsWrapper& self) {
    return fmt::format(
        "AudioPackets<id={}, src={}, timestamp=({}, {}), sample_format={}, {}>",
        self.get_packets()->id,
        self.get_packets()->src,
        std::get<0>(self.get_packets()->timestamp),
        std::get<1>(self.get_packets()->timestamp),
        self.get_packets()->get_media_format_name(),
        get_codec_info<MediaType::Audio>(self.get_packets()->codecpar));
  });

  _VideoPacketsWrapper.def("__repr__", [](const VideoPacketsWrapper& self) {
    return fmt::format(
        "VideoPackets<id={}, src={}, timestamp=({}, {}), frame_rate=({}/{}), num_packets={}, pixel_format={}, {}>",
        self.get_packets()->id,
        self.get_packets()->src,
        std::get<0>(self.get_packets()->timestamp),
        std::get<1>(self.get_packets()->timestamp),
        self.get_packets()->frame_rate.num,
        self.get_packets()->frame_rate.den,
        self.get_packets()->num_packets(),
        self.get_packets()->get_media_format_name(),
        get_codec_info<MediaType::Video>(self.get_packets()->codecpar));
  });

  _ImagePacketsWrapper.def("__repr__", [](const ImagePacketsWrapper& self) {
    return fmt::format(
        "ImagePackets<id={}, src={}, pixel_format={}, {}>",
        self.get_packets()->id,
        self.get_packets()->src,
        self.get_packets()->get_media_format_name(),
        get_codec_info<MediaType::Image>(self.get_packets()->codecpar));
  });
}
} // namespace spdl::core
