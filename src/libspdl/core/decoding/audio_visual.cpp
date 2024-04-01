#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/result.h"

#include <fmt/core.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <cstddef>
#include <cstdint>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {

template <MediaType media_type>
Task<FFmpegFramesWrapperPtr<media_type>> decode_task(
    PacketsPtr<media_type> packets,
    const DecodeConfig decode_cfg,
    const std::string filter_desc) {
  co_return wrap<media_type, FFmpegFramesPtr>(
      co_await detail::decode_packets_ffmpeg<media_type>(
          std::move(packets), std::move(decode_cfg), std::move(filter_desc)));
}

template <MediaType media_type>
Task<std::vector<SemiFuture<FFmpegFramesWrapperPtr<media_type>>>>
stream_decode_task(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const SourceAdaptorPtr adaptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  std::vector<SemiFuture<FFmpegFramesWrapperPtr<media_type>>> futures;
  {
    auto exec = detail::get_decode_executor(decode_executor);
    auto demuxer = detail::stream_demux<media_type>(
        src, timestamps, std::move(adaptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = decode_task<media_type>(
          *std::move(result), std::move(decode_cfg), std::move(filter_desc));
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return std::move(futures);
}
} // namespace

template <MediaType media_type>
Results<media_type, FFmpegFramesWrapperPtr> decoding::decode(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
  if (timestamps.size() == 0) {
    SPDL_FAIL("At least one timestamp must be provided.");
  }
  return Results<media_type, FFmpegFramesWrapperPtr>{
      new typename Results<media_type, FFmpegFramesWrapperPtr>::Impl{
          {src},
          timestamps,
          stream_decode_task<media_type>(
              src,
              timestamps,
              adaptor,
              io_cfg,
              decode_cfg,
              filter_desc,
              decode_executor)
              .scheduleOn(detail::get_demux_executor(demux_executor))
              .start()}};
}

template Results<MediaType::Audio, FFmpegFramesWrapperPtr> decoding::decode(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor);

template Results<MediaType::Video, FFmpegFramesWrapperPtr> decoding::decode(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor);

} // namespace spdl::core
