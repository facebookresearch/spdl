#include <libspdl/coro/decoding.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/decoding.h>
#include <libspdl/core/demuxing.h>

namespace spdl::coro {

template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = folly::coro::co_invoke(
      [=](PacketsPtr<media_type>&& pkts)
          -> folly::coro::Task<FFmpegFramesPtr<media_type>> {
        co_return decode_packets_ffmpeg(
            std::move(pkts), std::move(decode_cfg), std::move(filter_desc));
      },
      std::move(packets));
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

template FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<MediaType::Audio>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<MediaType::Audio> packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<MediaType::Video>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<MediaType::Video> packets,
    const std::optional<DecodeConfig> decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<MediaType::Image> packets,
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
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<FFmpegFramesPtr<MediaType::Image>> {
        co_return decode_packets_ffmpeg(
            demux_image(uri, adaptor, dmx_cfg),
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
    std::function<void(FFmpegFramesPtr<MediaType::Image>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<DecodeConfig>& decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear) {
  auto task = folly::coro::co_invoke(
      [=]() -> folly::coro::Task<FFmpegFramesPtr<MediaType::Image>> {
        co_return decode_packets_ffmpeg(
            demux_image(std::move(data), std::move(dmx_cfg), _zero_clear),
            std::move(decode_cfg),
            std::move(filter_desc));
      });

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

} // namespace spdl::coro
