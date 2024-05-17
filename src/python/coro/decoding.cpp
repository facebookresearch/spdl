#include <libspdl/coro/decoding.h>

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

namespace spdl::coro {

using spdl::core::AudioPacketsPtr;
using spdl::core::ImagePacketsPtr;
using spdl::core::VideoPacketsPtr;

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Async decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_sleep",
      &async_sleep,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("duration"),
      nb::arg("executor") = nullptr);

  m.def(
      "async_sleep_multi",
      &async_sleep_multi,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("duration"),
      nb::arg("count"),
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_audio",
      &async_decode<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_video",
      &async_decode<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_image",
      &async_decode<MediaType::Image>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr);

  m.def(
      "async_streaming_video_decoder",
      &async_streaming_decoder<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_video_frames",
      &async_decode_frames<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("decoder"),
      nb::arg("num_frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("executor") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Async demuxing + decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////

  m.def(
      "async_decode_image_from_source",
      &async_decode_from_source<MediaType::Image>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("_adaptor") = nullptr,
      nb::arg("demux_config") = nb::none(),
      nb::arg("decoder_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_image_from_bytes",
      &async_decode_from_bytes<MediaType::Image>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("decoder_config") = nb::none(),
      nb::arg("filter_desc") = "",
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_decode_video_nvdec",
      [](std::function<void(CUDABufferPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         VideoPacketsPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         const uintptr_t cuda_stream,
         const std::optional<cuda_allocator>& cuda_allocator,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Video>(
            set_result,
            notify_exception,
            std::move(packets),
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            cuda_stream,
            cuda_allocator,
            decode_executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
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
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);

  m.def(
      "async_decode_image_nvdec",
      [](std::function<void(CUDABufferPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         ImagePacketsPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         const uintptr_t cuda_stream,
         const std::optional<cuda_allocator>& cuda_allocator,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Image>(
            set_result,
            notify_exception,
            std::move(packets),
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            cuda_stream,
            cuda_allocator,
            decode_executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
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
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);

  m.def(
      "async_batch_decode_image_nvdec",
      [](std::function<void(CUDABufferPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::vector<ImagePacketsPtr>&& packets,
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
         const std::optional<cuda_allocator>& cuda_allocator,
         ThreadPoolExecutorPtr decode_executor) {
        return async_batch_decode_image_nvdec(
            set_result,
            notify_exception,
            std::move(packets),
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            strict,
            cuda_stream,
            cuda_allocator,
            decode_executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
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
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);
}
} // namespace spdl::coro
