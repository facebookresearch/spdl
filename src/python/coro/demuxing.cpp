#include <libspdl/coro/demuxing.h>

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
#include "libspdl/core/packets.h"

namespace nb = nanobind;

namespace spdl::coro {

using spdl::core::AudioPacketsPtr;
using spdl::core::ImagePacketsPtr;
using spdl::core::VideoPacketsPtr;

void register_demuxing(nb::module_& m) {
  m.def(
      "async_stream_demux_audio",
      &async_stream_demux<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamps") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_adaptor") = nullptr);

  m.def(
      "async_stream_demux_audio_bytes",
      [](std::function<void(AudioPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<DemuxConfig>& demux_config,
         ThreadPoolExecutorPtr executor,
         bool _zero_clear) {
        return async_stream_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamps,
            demux_config,
            executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamps") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_audio",
      &async_demux<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_adaptor") = nullptr);

  m.def(
      "async_demux_audio_bytes",
      [](std::function<void(AudioPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::optional<std::tuple<double, double>>& timestamp,
         const std::optional<DemuxConfig>& demux_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamp,
            demux_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_stream_demux_video",
      &async_stream_demux<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamps") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_adaptor") = nullptr);

  m.def(
      "async_stream_demux_video_bytes",
      [](std::function<void(VideoPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::optional<DemuxConfig>& demux_config,
         ThreadPoolExecutorPtr executor,
         bool _zero_clear) {
        return async_stream_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamps,
            demux_config,
            executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamps") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_video",
      &async_demux<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_adaptor") = nullptr);

  m.def(
      "async_demux_video_bytes",
      [](std::function<void(VideoPacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::optional<std::tuple<double, double>>& timestamp,
         const std::optional<DemuxConfig>& demux_config,
         ThreadPoolExecutorPtr executor,
         bool _zero_clear) {
        return async_demux_bytes<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            timestamp,
            demux_config,
            executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);

  m.def(
      "async_demux_image",
      &async_demux_image,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_adaptor") = nullptr);

  m.def(
      "async_demux_image_bytes",
      [](std::function<void(ImagePacketsPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         nb::bytes data,
         const std::optional<DemuxConfig>& demux_config,
         ThreadPoolExecutorPtr demux_executor,
         bool _zero_clear) {
        return async_demux_image_bytes(
            std::move(set_result),
            std::move(notify_exception),
            std::string_view{data.c_str(), data.size()},
            demux_config,
            demux_executor,
            _zero_clear);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("executor") = nullptr,
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::coro
