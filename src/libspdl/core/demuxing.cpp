#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
std::unique_ptr<Future> demux_async(
    std::function<void(std::optional<PacketsPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    std::shared_ptr<ThreadPoolExecutor> demux_executor) {
  return detail::execute_generator_with_callback<PacketsPtr<media_type>>(
      detail::stream_demux<media_type>(
          src, timestamps, std::move(adoptor), std::move(io_cfg)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

template std::unique_ptr<Future> demux_async<MediaType::Audio>(
    std::function<void(std::optional<PacketsPtr<MediaType::Audio>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    std::shared_ptr<ThreadPoolExecutor> demux_executor);

template std::unique_ptr<Future> demux_async<MediaType::Video>(
    std::function<void(std::optional<PacketsPtr<MediaType::Video>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    std::shared_ptr<ThreadPoolExecutor> demux_executor);

std::unique_ptr<Future> demux_image_async(
    std::function<void(std::optional<PacketsPtr<MediaType::Image>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    std::shared_ptr<ThreadPoolExecutor> demux_executor) {
  return detail::execute_task_with_callback<PacketsPtr<MediaType::Image>>(
      detail::demux_image(src, std::move(adoptor), std::move(io_cfg)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor(demux_executor));
}

} // namespace spdl::core
