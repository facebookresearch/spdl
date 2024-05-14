#include <libspdl/coro/demuxing.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/demuxing.h>

#include <folly/experimental/coro/AsyncGenerator.h>

namespace spdl::coro {

/// Demux audio or video
template <MediaType media_type>
FuturePtr async_stream_demux(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    SourceAdaptorPtr adaptor) {
  auto generator = folly::coro::co_invoke(
      [=]() -> folly::coro::AsyncGenerator<PacketsPtr<media_type>> {
        spdl::core::StreamingDemuxer<media_type> demuxer{uri, adaptor, dmx_cfg};
        for (auto& window : timestamps) {
          co_await folly::coro::co_safe_point;
          co_yield demuxer.demux_window(window);
        }
      });

  return detail::execute_generator_with_callback(
      std::move(generator),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_stream_demux(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr demux_executor,
    SourceAdaptorPtr adaptor);

template FuturePtr async_stream_demux(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr demux_executor,
    SourceAdaptorPtr adaptor);

template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    SourceAdaptorPtr adaptor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<PacketsPtr<media_type>> {
        spdl::core::StreamingDemuxer<media_type> demuxer{uri, adaptor, dmx_cfg};
        co_return demuxer.demux_window(timestamp);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr demux_executor,
    SourceAdaptorPtr adaptor);

template FuturePtr async_demux(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr demux_executor,
    SourceAdaptorPtr adaptor);

template <MediaType media_type>
FuturePtr async_stream_demux_bytes(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto generator = folly::coro::co_invoke(
      [=]() -> folly::coro::AsyncGenerator<PacketsPtr<media_type>> {
        spdl::core::StreamingDemuxer<media_type> demuxer{data, dmx_cfg};
        for (auto& window : timestamps) {
          co_await folly::coro::co_safe_point;
          co_yield demuxer.demux_window(window);
        }
        if (_zero_clear) {
          std::memset((void*)data.data(), 0, data.size());
        }
      });

  return detail::execute_generator_with_callback(
      std::move(generator),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_stream_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

template FuturePtr async_stream_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<PacketsPtr<media_type>> {
        spdl::core::StreamingDemuxer<media_type> demuxer{data, dmx_cfg};
        auto ret = demuxer.demux_window(timestamp);
        if (_zero_clear) {
          std::memset((void*)data.data(), 0, data.size());
        }
        co_return ret;
      });

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

template FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

FuturePtr async_demux_image(
    std::function<void(PacketsPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    SourceAdaptorPtr adaptor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<PacketsPtr<MediaType::Image>> {
        co_return demux_image(
            std::move(uri), std::move(adaptor), std::move(dmx_cfg));
      });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

FuturePtr async_demux_image_bytes(
    std::function<void(PacketsPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<PacketsPtr<MediaType::Image>> {
        co_return demux_image(std::move(data), std::move(dmx_cfg), _zero_clear);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}
} // namespace spdl::coro
