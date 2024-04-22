#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"

namespace spdl::core {

template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsWrapperPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_cfg,
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
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegVideoFramesWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    VideoPacketsWrapperPtr packets,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    ImagePacketsWrapperPtr packets,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

//////////////////////////////////////////////////////////////////////////////
// Demuxing + decoding in one go
//////////////////////////////////////////////////////////////////////////////

template <>
FuturePtr async_decode_from_source(
    std::function<void(FFmpegFramesWrapperPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const SourceAdaptorPtr& adaptor,
    const std::optional<IOConfig>& io_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<FFmpegFramesWrapperPtr<MediaType::Image>> {
        auto frames = co_await detail::decode_packets_ffmpeg<MediaType::Image>(
            co_await detail::demux_image(uri, adaptor, io_cfg),
            std::move(decode_cfg),
            std::move(filter_desc));
        co_return wrap<MediaType::Image, FFmpegFramesPtr>(std::move(frames));
      });

  return detail::execute_task_with_callback<
      FFmpegFramesWrapperPtr<MediaType::Image>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template <>
FuturePtr async_decode_from_bytes(
    std::function<void(FFmpegFramesWrapperPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string_view data,
    const std::optional<IOConfig>& io_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<FFmpegFramesWrapperPtr<MediaType::Image>> {
        auto packets = co_await detail::demux_image(
            data,
            std::unique_ptr<SourceAdaptor>(new BytesAdaptor()),
            std::move(io_cfg));
        if (_zero_clear) {
          std::memset((void*)data.data(), 0, data.size());
        }
        auto frames = co_await detail::decode_packets_ffmpeg<MediaType::Image>(
            std::move(packets), std::move(decode_cfg), std::move(filter_desc));
        co_return wrap<MediaType::Image, FFmpegFramesPtr>(std::move(frames));
      });

  return detail::execute_task_with_callback<
      FFmpegFramesWrapperPtr<MediaType::Image>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

} // namespace spdl::core
