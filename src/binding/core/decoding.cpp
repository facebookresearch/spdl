#include <libspdl/core/decoding.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
FFmpegFramesPtr<media_type> decode(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> cfg,
    const std::string filter_desc) {
  nb::gil_scoped_release g;
  return decode_packets_ffmpeg(std::move(packets), cfg, filter_desc);
}

template <MediaType media_type>
CUDABufferPtr decode_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig cuda_config,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt) {
  nb::gil_scoped_release g;
  return decode_packets_nvdec(
      std::move(packets),
      cuda_config,
      CropArea{
          static_cast<short>(crop_left),
          static_cast<short>(crop_top),
          static_cast<short>(crop_right),
          static_cast<short>(crop_bottom)},
      width,
      height,
      pix_fmt);
}

template <MediaType media_type>
DecoderPtr<media_type> _make_decoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::string& filter_desc) {
  nb::gil_scoped_release g;
  return make_decoder(
      std::move(packets), std::move(decode_cfg), std::move(filter_desc));
}

template <MediaType media_type>
void _drop(DecoderPtr<media_type> decoder) {
  nb::gil_scoped_release g;
  decoder.reset();
}

template <MediaType media_type>
std::tuple<DecoderPtr<media_type>, std::optional<FFmpegFramesPtr<media_type>>>
_decode(DecoderPtr<media_type> decoder, int num_frames) {
  nb::gil_scoped_release g;
  auto frames = decoder->decode(num_frames);
  return {std::move(decoder), std::move(frames)};
}

void zero_clear(nb::bytes data) {
  std::memset((void*)data.c_str(), 0, data.size());
}

} // namespace

void register_decoding(nb::module_& m) {
  nb::class_<StreamingDecoder<MediaType::Video>>(m, "StreamingVideoDecoder");

  m.def(
      "_streaming_decoder",
      &_make_decoder<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "");

  m.def("_decode", &_decode<MediaType::Video>);
  m.def("_drop", &_drop<MediaType::Video>);

  ////////////////////////////////////////////////////////////////////////////////
  // Async decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_packets",
      &decode<MediaType::Audio>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "");

  m.def(
      "decode_packets",
      &decode<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "");

  m.def(
      "decode_packets",
      &decode<MediaType::Image>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "");

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_packets_nvdec",
      &decode_nvdec<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_config"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba");

  m.def(
      "decode_packets_nvdec",
      &decode_nvdec<MediaType::Image>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_config"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba");

  m.def(
      "decode_packets_nvdec",
      [](std::vector<ImagePacketsPtr>&& packets,
         const CUDAConfig& cuda_config,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         bool strict) {
        nb::gil_scoped_release g;
        return decode_packets_nvdec(
            std::move(packets),
            cuda_config,
            CropArea{
                static_cast<short>(crop_left),
                static_cast<short>(crop_top),
                static_cast<short>(crop_right),
                static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            strict);
      },
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_config"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba",
      nb::arg("strict") = true);

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVJPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_image_nvjpeg",
      [](nb::bytes data,
         const CUDAConfig cuda_config,
         int scale_width,
         int scale_height,
         const std::string& pix_fmt,
         bool _zero_clear) {
        nb::gil_scoped_release g;
        auto ret = decode_image_nvjpeg(
            std::string_view{data.c_str(), data.size()},
            cuda_config,
            scale_width,
            scale_height,
            pix_fmt);
        if (_zero_clear) {
          nb::gil_scoped_acquire gg;
          zero_clear(data);
        }
        return ret;
      },
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_config"),
      nb::arg("scale_width") = -1,
      nb::arg("scale_height") = -1,
      nb::arg("pix_fmt") = "rgb",
      nb::arg("_zero_clear") = false);

  m.def(
      "decode_image_nvjpeg",
      [](const std::vector<nb::bytes>& data,
         const CUDAConfig cuda_config,
         int scale_width,
         int scale_height,
         const std::string& pix_fmt,
         bool _zero_clear) {
        std::vector<std::string_view> dataset;
        for (const auto& d : data) {
          dataset.push_back(std::string_view{d.c_str(), d.size()});
        }
        nb::gil_scoped_release g;
        auto ret = decode_image_nvjpeg(
            dataset, cuda_config, scale_width, scale_height, pix_fmt);
        if (_zero_clear) {
          nb::gil_scoped_acquire gg;
          for (auto& d : data) {
            zero_clear(d);
          }
        }
        return ret;
      },
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_config"),
      nb::arg("scale_width"),
      nb::arg("scale_height"),
      nb::arg("pix_fmt") = "rgb",
      nb::arg("_zero_clear") = false);
}

} // namespace spdl::core
