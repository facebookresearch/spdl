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
      "async_apply_bsf",
      &async_apply_bsf,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_audio",
      [](std::function<void(std::optional<AudioPacketsWrapperPtr>)> set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("demux_executor") = nullptr);

  m.def(
      "async_demux_video",
      [](std::function<void(std::optional<VideoPacketsWrapperPtr>)> set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("demux_executor") = nullptr);

  m.def(
      "async_demux_image",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux_image(
            std::move(set_result),
            std::move(notify_exception),
            src,
            adoptor,
            {format, format_options, buffer_size},
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("demux_executor") = nullptr);
}
} // namespace spdl::core
