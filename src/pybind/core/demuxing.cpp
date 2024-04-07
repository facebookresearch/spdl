#include <libspdl/core/adaptor.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/result.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {

void register_demuxing(py::module& m) {
  m.def(
      "async_demux_audio",
      [](std::function<void(AudioPacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::str src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdaptorPtr& adaptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         const std::optional<int>& buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            timestamps,
            adaptor,
            {format,
             format_options,
             buffer_size.value_or(SPDL_DEFAULT_BUFFER_SIZE)},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_audio_bytes",
      [](std::function<void(AudioPacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            timestamps,
            {format, format_options, buffer_size},
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_video",
      [](std::function<void(VideoPacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::str src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdaptorPtr& adaptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            timestamps,
            adaptor,
            {format, format_options, buffer_size},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_video_bytes",
      [](std::function<void(VideoPacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            timestamps,
            {format, format_options, buffer_size},
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_image",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::str src,
         const SourceAdaptorPtr& adaptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux_image(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            adaptor,
            {format, format_options, buffer_size},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_image_bytes",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void(std::string)> notify_exception,
         py::bytes data,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            {format, format_options, buffer_size},
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::kw_only(),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);
}
} // namespace spdl::core
