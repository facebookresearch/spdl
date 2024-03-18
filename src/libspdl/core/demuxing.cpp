#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

/// Demux audio or video
template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(std::optional<PacketsWrapperPtr<media_type>>)>
        set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::AsyncGenerator<PacketsWrapperPtr<media_type>> {
        auto generator = detail::stream_demux<media_type>(
            src, timestamps, std::move(adoptor), std::move(io_cfg));
        while (auto result = co_await generator.next()) {
          co_yield wrap(std::move(*result));
        }
      });
  return detail::execute_generator_with_callback<PacketsWrapperPtr<media_type>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

template FuturePtr async_demux(
    std::function<void(std::optional<PacketsWrapperPtr<MediaType::Audio>>)>
        set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_demux(
    std::function<void(std::optional<PacketsWrapperPtr<MediaType::Video>>)>
        set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

FuturePtr async_demux_image(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<ImagePacketsWrapperPtr> {
        auto result = co_await detail::demux_image(
            src, std::move(adoptor), std::move(io_cfg));
        co_return wrap(std::move(result));
      });
  return detail::execute_task_with_callback<ImagePacketsWrapperPtr>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

FuturePtr async_apply_bsf(
    std::function<void(VideoPacketsWrapperPtr)> set_result,
    std::function<void()> notify_exception,
    VideoPacketsWrapperPtr packets,
    ThreadPoolExecutorPtr demux_executor) {
  auto task = folly::coro::co_invoke(
      [=](VideoPacketsPtr&& pkts) -> folly::coro::Task<VideoPacketsWrapperPtr> {
        auto filtered = co_await detail::apply_bsf(std::move(pkts));
        co_return wrap<MediaType::Video>(std::move(filtered));
      },
      packets->unwrap());
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor(demux_executor));
}

} // namespace spdl::core
