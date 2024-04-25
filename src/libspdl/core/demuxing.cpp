#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

/// Demux audio or video
template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor) {
  return detail::execute_generator_with_callback(
      detail::stream_demux<media_type>(
          std::move(uri),
          std::move(adaptor),
          std::move(io_cfg),
          std::move(timestamps)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_demux(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::vector<std::tuple<double, double>> timestamps,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  return detail::execute_generator_with_callback(
      detail::stream_demux<media_type>(
          std::move(data),
          std::move(io_cfg),
          std::move(timestamps),
          _zero_clear),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::vector<std::tuple<double, double>> timestamps,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

template FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::vector<std::tuple<double, double>> timestamps,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

FuturePtr async_demux_image(
    std::function<void(ImagePacketsPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    SourceAdaptorPtr adaptor,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor) {
  return detail::execute_task_with_callback(
      detail::demux_image(
          std::move(uri), std::move(adaptor), std::move(io_cfg)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

FuturePtr async_demux_image_bytes(
    std::function<void(ImagePacketsPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::optional<IOConfig> io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  return detail::execute_task_with_callback(
      detail::demux_image(std::move(data), std::move(io_cfg), _zero_clear),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}
} // namespace spdl::core
