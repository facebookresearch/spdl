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
      "demux_audio_async",
      [](std::function<void(std::optional<PacketsPtr<MediaType::Audio>>)>
             set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return demux_async<MediaType::Audio>(
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
      "demux_video_async",
      [](std::function<void(std::optional<PacketsPtr<MediaType::Video>>)>
             set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return demux_async<MediaType::Video>(
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
      "demux_image_async",
      [](std::function<void(std::optional<PacketsPtr<MediaType::Image>>)>
             set_result,
         std::function<void()> notify_exception,
         const std::string& src,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         ThreadPoolExecutorPtr demux_executor) {
        return demux_image_async(
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
