#include <libspdl/core/adaptor.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
using IOConfigPtr = std::shared_ptr<spdl::core::IOConfig>;

namespace {
IOConfigPtr make_io_config(
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options,
    const std::optional<int>& buffer_size) {
  auto ret = std::make_shared<spdl::core::IOConfig>();
  ret->format = format;
  ret->format_options = format_options;
  if (buffer_size) {
    ret->buffer_size = buffer_size.value();
  }
  return ret;
}

} // namespace

void register_demuxing(py::module& m) {
  auto _IOConfig =
      py::class_<IOConfig, IOConfigPtr>(m, "IOConfig", py::module_local());

  _IOConfig.def(
      py::init(&make_io_config),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = py::none());

  m.def(
      "async_demux_audio",
      [](std::function<void(AudioPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::str src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdaptorPtr& adaptor,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            timestamps,
            adaptor,
            io_config,
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_audio_bytes",
      [](std::function<void(AudioPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_audio_buffer",
      [](std::function<void(AudioPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::buffer data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        auto buffer_info = data.request(/*writable=*/_zero_clear);
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{(char*)buffer_info.ptr, (size_t)buffer_info.size},
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_video",
      [](std::function<void(VideoPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::str src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdaptorPtr& adaptor,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            timestamps,
            adaptor,
            io_config,
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_video_bytes",
      [](std::function<void(VideoPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_video_buffer",
      [](std::function<void(VideoPacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::buffer data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        auto buffer_info = data.request(/*writable=*/_zero_clear);
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{(char*)buffer_info.ptr, (size_t)buffer_info.size},
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_image",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::str src,
         const SourceAdaptorPtr& adaptor,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor) {
        return async_demux_image(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string>(src),
            adaptor,
            io_config,
            demux_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("src"),
      py::kw_only(),
      py::arg("adaptor") = nullptr,
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_demux_image_bytes",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::bytes data,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            static_cast<std::string_view>(data),
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);

  m.def(
      "async_demux_image_buffer",
      [](std::function<void(ImagePacketsWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         py::buffer data,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        auto buffer_info = data.request(/*writable=*/_zero_clear);
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{(char*)buffer_info.ptr, (size_t)buffer_info.size},
            io_config,
            demux_executor,
            _zero_clear);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("data"),
      py::kw_only(),
      py::arg("io_config") = py::none(),
      py::arg("executor") = nullptr,
      py::arg("_zero_clear") = false);
}
} // namespace spdl::core
