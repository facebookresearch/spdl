#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

/// Demux audio or video
template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(PacketsWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::AsyncGenerator<PacketsWrapperPtr<media_type>> {
        auto generator = detail::stream_demux<media_type>(
            src, timestamps, std::move(adaptor), std::move(io_cfg));
        while (auto result = co_await generator.next()) {
          co_yield wrap(std::move(*result));
        }
      });
  return detail::execute_generator_with_callback<PacketsWrapperPtr<media_type>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux(
    std::function<void(PacketsWrapperPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_demux(
    std::function<void(PacketsWrapperPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(PacketsWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::AsyncGenerator<PacketsWrapperPtr<media_type>> {
        auto generator = detail::stream_demux<media_type>(
            data,
            timestamps,
            std::unique_ptr<SourceAdaptor>(new BytesAdaptor()),
            std::move(io_cfg));
        while (auto result = co_await generator.next()) {
          co_yield wrap(std::move(*result));
        }
        if (_zero_clear) {
          std::memset((void*)data.data(), 0, data.size());
        }
      });
  return detail::execute_generator_with_callback<PacketsWrapperPtr<media_type>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

template FuturePtr async_demux_bytes(
    std::function<void(PacketsWrapperPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

template FuturePtr async_demux_bytes(
    std::function<void(PacketsWrapperPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear);

FuturePtr async_demux_image(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<ImagePacketsWrapperPtr> {
        auto result = co_await detail::demux_image(
            src, std::move(adaptor), std::move(io_cfg));
        co_return wrap(std::move(result));
      });
  return detail::execute_task_with_callback<ImagePacketsWrapperPtr>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}

FuturePtr async_demux_image_bytes(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::optional<IOConfig>& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<ImagePacketsWrapperPtr> {
        auto result = co_await detail::demux_image(
            data,
            std::unique_ptr<SourceAdaptor>(new BytesAdaptor()),
            std::move(io_cfg));
        if (_zero_clear) {
          std::memset((void*)data.data(), 0, data.size());
        }
        co_return wrap(std::move(result));
      });
  return detail::execute_task_with_callback<ImagePacketsWrapperPtr>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(executor));
}
} // namespace spdl::core
