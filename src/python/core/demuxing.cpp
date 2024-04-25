#include <libspdl/core/adaptor.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
using IOConfigPtr = std::shared_ptr<spdl::core::IOConfig>;

void register_demuxing(nb::module_& m) {
  nb::class_<IOConfig>(m, "IOConfig")
      .def(
          nb::init<
              const std::optional<std::string>,
              const std::optional<OptionDict>,
              int>(),
          nb::arg("format") = nb::none(),
          nb::arg("format_options") = nb::none(),
          nb::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE);

  m.def(
      "async_demux_audio",
      &async_demux<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("_adaptor") = nullptr,
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr);

  m.def(
      "async_demux_audio_bytes",
      [](std::function<void(AudioPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_audio_buffer",
      [](std::function<void(AudioPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::string_view data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        // auto buffer_info = data.request(/*writable=*/_zero_clear);
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            data,
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_video",
      [](std::function<void(VideoPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         const SourceAdaptorPtr& _adaptor,
         ThreadPoolExecutorPtr _executor) {
        return async_demux<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            src,
            timestamps,
            _adaptor,
            io_config,
            _executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_adaptor") = nullptr,
      nb::arg("_executor") = nullptr);

  m.def(
      "async_demux_video_bytes",
      [](std::function<void(VideoPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_video_buffer",
      [](std::function<void(VideoPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::string_view data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            data,
            timestamps,
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      nb::arg("timestamps"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_image",
      [](std::function<void(ImagePacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         const std::string& src,
         const std::optional<IOConfig>& io_config,
         const SourceAdaptorPtr& _adaptor,
         ThreadPoolExecutorPtr _executor) {
        return async_demux_image(
            std::move(set_result),
            std::move(notify_exception),
            src,
            _adaptor,
            io_config,
            _executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_adaptor") = nullptr,
      nb::arg("_executor") = nullptr);

  m.def(
      "async_demux_image_bytes",
      [](std::function<void(ImagePacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_image_buffer",
      [](std::function<void(ImagePacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::string_view data,
         const std::optional<IOConfig>& io_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            data,
            io_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
      // nb::kw_only(),
      nb::arg("io_config") = nb::none(),
      nb::arg("_executor") = nullptr,
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::core
