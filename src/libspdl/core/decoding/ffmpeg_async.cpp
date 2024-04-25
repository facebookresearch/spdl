#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  return detail::execute_task_with_callback(
      detail::decode_packets_ffmpeg(
          std::move(packets), std::move(decode_cfg), std::move(filter_desc)),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template FuturePtr async_decode(
    std::function<void(FFmpegAudioFramesPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    AudioPacketsPtr packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegVideoFramesPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    VideoPacketsPtr packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegImageFramesPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    ImagePacketsPtr packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

//////////////////////////////////////////////////////////////////////////////
// Demuxing + decoding in one go
//////////////////////////////////////////////////////////////////////////////

template <>
FuturePtr async_decode_from_source(
    std::function<void(FFmpegFramesPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<FFmpegImageFramesPtr> {
        co_return co_await detail::decode_packets_ffmpeg(
            co_await detail::demux_image(uri, adaptor, io_cfg),
            std::move(decode_cfg),
            std::move(filter_desc));
      });

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template <>
FuturePtr async_decode_from_bytes(
    std::function<void(FFmpegImageFramesPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string_view data,
    const std::optional<IOConfig>& io_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<FFmpegImageFramesPtr> {
        co_return co_await detail::decode_packets_ffmpeg(
            co_await detail::demux_image(
                std::move(data), std::move(io_cfg), _zero_clear),
            std::move(decode_cfg),
            std::move(filter_desc));
      });

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

} // namespace spdl::core
