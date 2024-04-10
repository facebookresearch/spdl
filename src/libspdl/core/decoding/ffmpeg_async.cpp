#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsWrapperPtr<media_type> packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = folly::coro::co_invoke(
      [=](PacketsPtr<media_type> pkts)
          -> folly::coro::Task<FFmpegFramesWrapperPtr<media_type>> {
        auto frames = co_await detail::decode_packets_ffmpeg<media_type>(
            std::move(pkts), std::move(decode_cfg), std::move(filter_desc));
        co_return wrap<media_type, FFmpegFramesPtr>(std::move(frames));
      },
      packets->unwrap());

  return detail::execute_task_with_callback<FFmpegFramesWrapperPtr<media_type>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template FuturePtr async_decode(
    std::function<void(FFmpegAudioFramesWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    AudioPacketsWrapperPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegVideoFramesWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    VideoPacketsWrapperPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    ImagePacketsWrapperPtr packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

} // namespace spdl::core
