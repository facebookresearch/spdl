#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
FuturePtr decode_async(
    std::function<void(std::optional<FFmpegFramesPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    PacketsPtr<media_type> packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  return detail::execute_task_with_callback<FFmpegFramesPtr<media_type>>(
      detail::decode_packets_ffmpeg<media_type>(
          std::move(packets), std::move(decode_cfg), std::move(filter_desc)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template FuturePtr decode_async(
    std::function<void(std::optional<FFmpegAudioFramesPtr>)>
        set_result,
    std::function<void()> notify_exception,
    AudioPacketsPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr decode_async(
    std::function<void(std::optional<FFmpegVideoFramesPtr>)>
        set_result,
    std::function<void()> notify_exception,
    VideoPacketsPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr decode_async(
    std::function<void(std::optional<FFmpegImageFramesPtr>)>
        set_result,
    std::function<void()> notify_exception,
    ImagePacketsPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

} // namespace spdl::core
