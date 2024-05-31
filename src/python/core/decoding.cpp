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
    const int cuda_device_index,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator) {
  nb::gil_scoped_release g;
  return decode_packets_nvdec(
      std::move(packets),
      cuda_device_index,
      CropArea{
          static_cast<short>(crop_left),
          static_cast<short>(crop_top),
          static_cast<short>(crop_right),
          static_cast<short>(crop_bottom)},
      width,
      height,
      pix_fmt,
      cuda_stream,
      cuda_allocator);
}

template <MediaType media_type>
std::shared_ptr<StreamingDecoder<media_type>> make_decoder(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::string& filter_desc) {
  nb::gil_scoped_release g;
  return std::make_shared<spdl::core::StreamingDecoder<media_type>>(
      std::move(packets), std::move(decode_cfg), std::move(filter_desc));
}

template <MediaType media_type>
std::optional<FFmpegFramesPtr<media_type>> decoder_decode(
    std::shared_ptr<StreamingDecoder<media_type>> self,
    int num_frames) {
  nb::gil_scoped_release g;
  return self->decode(num_frames);
}

} // namespace

void register_decoding(nb::module_& m) {
  nb::class_<StreamingDecoder<MediaType::Video>>(m, "StreamingVideoDecoder");

  m.def(
      "_streaming_decoder",
      &make_decoder<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "");

  m.def("_decode", &decoder_decode<MediaType::Video>);

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
      nb::arg("cuda_device_index"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba",
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "decode_packets_nvdec",
      &decode_nvdec<MediaType::Image>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba",
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "decode_packets_nvdec",
      [](std::vector<ImagePacketsPtr>&& packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         bool strict,
         const uintptr_t cuda_stream,
         const std::optional<cuda_allocator>& cuda_allocator) {
        nb::gil_scoped_release g;
        return decode_packets_nvdec(
            std::move(packets),
            cuda_device_index,
            CropArea{
                static_cast<short>(crop_left),
                static_cast<short>(crop_top),
                static_cast<short>(crop_right),
                static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            strict,
            cuda_stream,
            cuda_allocator);
      },
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("crop_left") = 0,
      nb::arg("crop_top") = 0,
      nb::arg("crop_right") = 0,
      nb::arg("crop_bottom") = 0,
      nb::arg("width") = -1,
      nb::arg("height") = -1,
      nb::arg("pix_fmt").none() = "rgba",
      nb::arg("strict") = true,
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVJPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_image_nvjpeg",
      [](nb::bytes data,
         int cuda_device_index,
         const std::string& pix_fmt,
         const std::optional<cuda_allocator>& cuda_allocator) {
        nb::gil_scoped_release g;
        return decode_image_nvjpeg(
            std::string_view{data.c_str(), data.size()},
            cuda_device_index,
            pix_fmt,
            cuda_allocator);
      },
      nb::arg("data"),
      nb::arg("cuda_device_index"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("pix_fmt") = "rgb",
      nb::arg("cuda_allocator") = nb::none());
}

} // namespace spdl::core
