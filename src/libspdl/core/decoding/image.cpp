#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/result.h"

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {
Task<FFmpegImageFramesWrapperPtr> image_decode_task(
    const std::string src,
    const SourceAdoptorPtr adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor) {
  co_return wrap<MediaType::Image, FFmpegFramesPtr>(
      co_await detail::decode_packets_ffmpeg<MediaType::Image>(
          co_await detail::demux_image(
              src, std::move(adoptor), std::move(io_cfg)),
          std::move(decode_cfg),
          std::move(filter_desc))
          .scheduleOn(detail::get_decode_executor(decode_executor)));
}

Task<std::vector<SemiFuture<FFmpegImageFramesWrapperPtr>>>
batch_image_decode_task(
    const std::vector<std::string> srcs,
    const SourceAdoptorPtr adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
  std::vector<SemiFuture<FFmpegImageFramesWrapperPtr>> futures;
  for (auto& src : srcs) {
    futures.emplace_back(
        image_decode_task(
            src, adoptor, io_cfg, decode_cfg, filter_desc, decode_executor)
            .scheduleOn(detail::get_demux_executor(demux_executor))
            .start());
  }
  co_return std::move(futures);
}
} // namespace

DecodeResult<MediaType::Image> decoding::decode_image(
    const std::string& src,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = image_decode_task(
      src, adoptor, io_cfg, decode_cfg, filter_desc, decode_executor);
  return DecodeResult<MediaType::Image>{
      new DecodeResult<MediaType::Image>::Impl{
          std::move(task)
              .scheduleOn(detail::get_demux_executor(demux_executor))
              .start()}};
}

BatchDecodeResult<MediaType::Image> decoding::batch_decode_image(
    const std::vector<std::string>& srcs,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
  if (srcs.size() == 0) {
    SPDL_FAIL("At least one image source must be provided.");
  }
  return BatchDecodeResult<MediaType::Image>{
      new BatchDecodeResult<MediaType::Image>::Impl{
          srcs,
          {},
          batch_image_decode_task(
              srcs,
              adoptor,
              io_cfg,
              decode_cfg,
              filter_desc,
              demux_executor,
              decode_executor)
              .scheduleOn(detail::get_demux_executor(demux_executor))
              .start()}};
}
} // namespace spdl::core
