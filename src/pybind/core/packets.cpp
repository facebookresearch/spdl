#include <libspdl/core/packets.h>

#include <pybind11/pybind11.h>

#include <fmt/format.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
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
    AVSampleFormat smp_fmt = static_cast<AVSampleFormat>(codecpar->format);
    parts.emplace_back(fmt::format(
        "sample_format: {}",
        (smp_fmt == AV_SAMPLE_FMT_NONE) ? "unknown"
                                        : av_get_sample_fmt_name(smp_fmt)));
    parts.emplace_back(fmt::format("sample_rate: {}", codecpar->sample_rate));
    parts.emplace_back(fmt::format("num_channels: {}", codecpar->channels));
  }
  if constexpr (
      media_type == MediaType::Video || media_type == MediaType::Image) {
    AVPixelFormat pix_fmt = static_cast<AVPixelFormat>(codecpar->format);
    parts.emplace_back(fmt::format(
        "pixel_format: {}",
        (pix_fmt == AV_PIX_FMT_NONE) ? "unknown"
                                     : av_get_pix_fmt_name(pix_fmt)));
    parts.emplace_back(fmt::format(
        "width: {}, height: {}", codecpar->width, codecpar->height));
  }
  return fmt::format("{}", fmt::join(parts, ", "));
}

} // namespace

void register_packets(py::module& m) {
  auto _AudioPackets = py::class_<AudioPackets, AudioPacketsPtr>(
      m, "AudioPackets", py::module_local());
  auto _VideoPackets = py::class_<VideoPackets, VideoPacketsPtr>(
      m, "VideoPackets", py::module_local());
  auto _ImagePackets = py::class_<ImagePackets, ImagePacketsPtr>(
      m, "ImagePackets", py::module_local());

  _AudioPackets.def("__repr__", [](const AudioPackets& self) {
    return fmt::format(
        "AudioPackets<id={}, src={}, timestamp=({}, {}), {}>",
        self.id,
        self.src,
        std::get<0>(self.timestamp),
        std::get<1>(self.timestamp),
        get_codec_info<MediaType::Audio>(self.codecpar));
  });

  _VideoPackets.def("__repr__", [](const VideoPackets& self) {
    return fmt::format(
        "VideoPackets<id={}, src={}, timestamp=({}, {}), frame_rate=({}/{}), num_packets={}, {}>",
        self.id,
        self.src,
        std::get<0>(self.timestamp),
        std::get<1>(self.timestamp),
        self.frame_rate.num,
        self.frame_rate.den,
        self.num_packets(),
        get_codec_info<MediaType::Video>(self.codecpar));
  });

  _ImagePackets.def("__repr__", [](const ImagePackets& self) {
    return fmt::format(
        "ImagePackets<id={}, src={}, {}>",
        self.id,
        self.src,
        get_codec_info<MediaType::Image>(self.codecpar));
  });
}
} // namespace spdl::core
