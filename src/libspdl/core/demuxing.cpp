#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(std::optional<PacketsPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor) {
  return detail::execute_generator_with_callback<PacketsPtr<media_type>>(
      detail::stream_demux<media_type>(
          src, timestamps, std::move(adoptor), std::move(io_cfg)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

template FuturePtr async_demux<MediaType::Audio>(
    std::function<void(std::optional<AudioPacketsPtr>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_demux<MediaType::Video>(
    std::function<void(std::optional<VideoPacketsPtr>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

FuturePtr async_demux_image(
    std::function<void(std::optional<ImagePacketsPtr>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor) {
  return detail::execute_task_with_callback<ImagePacketsPtr>(
      detail::demux_image(src, std::move(adoptor), std::move(io_cfg)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

FuturePtr async_apply_bsf(
    std::function<void(VideoPacketsPtr)> set_result,
    std::function<void()> notify_exception,
    VideoPacketsPtr packets,
    ThreadPoolExecutorPtr demux_executor) {
  return detail::execute_task_with_callback(
      detail::apply_bsf(std::move(packets)),
      set_result,
      notify_exception,
      detail::get_demux_executor(demux_executor));
}

} // namespace spdl::core
