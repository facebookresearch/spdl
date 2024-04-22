#include <libspdl/core/decoding.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>

extern "C" {
#include <libavfilter/version.h>
}

namespace py = pybind11;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
using DecodeConfigPtr = std::shared_ptr<spdl::core::DecodeConfig>;

DecodeConfigPtr make_decode_config(
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_options) {
  auto ret = std::make_shared<spdl::core::DecodeConfig>();
  ret->decoder = decoder;
  ret->decoder_options = decoder_options;
  return ret;
}

std::optional<Rational> get_frame_rate(const py::object& frame_rate) {
  if (frame_rate.is(py::none())) {
    return std::nullopt;
  }
  py::object Fraction = py::module_::import("fractions").attr("Fraction");
  py::object r = Fraction(frame_rate);
  return {Rational{
      r.attr("numerator").cast<int>(), r.attr("denominator").cast<int>()}};
}
} // namespace

void register_decoding(py::module& m) {
  auto _DecodeConfig = py::class_<DecodeConfig, DecodeConfigPtr>(
      m, "DecodeConfig", py::module_local());

  _DecodeConfig.def(
      py::init(&make_decode_config),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none());

  ////////////////////////////////////////////////////////////////////////////////
  // Async decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_sleep",
      &async_sleep,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("duration"),
      py::arg("_executor") = nullptr);

  m.def(
      "async_sleep_multi",
      &async_sleep_multi,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("duration"),
      py::arg("count"),
      py::arg("_executor") = nullptr);

  m.def(
      "async_decode_audio",
      &async_decode<MediaType::Audio>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decode_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr);

  m.def(
      "async_decode_video",
      &async_decode<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decode_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr);

  m.def(
      "async_decode_image",
      &async_decode<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decode_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Async demuxing + decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////

  m.def(
      "async_decode_image_from_source",
      &async_decode_from_source<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::kw_only(),
      py::arg("_adaptor") = nullptr,
      py::arg("io_config") = py::none(),
      py::arg("decoder_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr);

  m.def(
      "async_decode_image_from_bytes",
      &async_decode_from_bytes<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("decoder_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_decode_image_from_buffer",
      [](std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::buffer data,
         const std::optional<IOConfig>& io_config,
         const std::optional<DecodeConfig>& decode_config,
         std::string filter_desc,
         std::shared_ptr<ThreadPoolExecutor> _executor,
         bool _zero_clear) {
        auto buffer_info = data.request(/*writable=*/_zero_clear);
        return async_decode_from_bytes<MediaType::Image>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{(char*)buffer_info.ptr, (size_t)buffer_info.size},
            io_config,
            decode_config,
            std::move(filter_desc),
            _executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("decoder_config") = py::none(),
      py::arg("filter_desc") = "",
      py::arg("_executor") = nullptr,
      py::arg("_zero_clear") = false);

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_decode_video_nvdec",
      [](std::function<void(NvDecVideoFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         VideoPacketsWrapperPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Video>(
            set_result,
            notify_exception,
            packets,
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("cuda_device_index"),
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba",
      py::arg("_executor") = nullptr);

  m.def(
      "async_decode_image_nvdec",
      [](std::function<void(NvDecImageFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         ImagePacketsWrapperPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Image>(
            set_result,
            notify_exception,
            packets,
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("cuda_device_index"),
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba",
      py::arg("_executor") = nullptr);
}
} // namespace spdl::core
