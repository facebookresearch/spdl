#include <libspdl/core/decoding.h>

#include <libspdl/core/detail/executor.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/logging.h>
#include <libspdl/core/detail/result.h>

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {
Task<std::unique_ptr<DecodedFrames>> image_decode_task(
    const std::string src,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  co_return co_await detail::decode_packets(
      co_await detail::demux_image(src, std::move(adoptor), std::move(io_cfg)),
      std::move(decode_cfg),
      std::move(filter_desc))
      .scheduleOn(detail::get_decode_executor(decode_executor));
}

Task<std::vector<SemiFuture<std::unique_ptr<DecodedFrames>>>>
batch_image_decode_task(
    const std::vector<std::string> srcs,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  std::vector<SemiFuture<std::unique_ptr<DecodedFrames>>> futures;
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

DecodeImageResult decoding::async_decode_image(
    const std::string& src,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  auto task = image_decode_task(
      src, adoptor, io_cfg, decode_cfg, filter_desc, decode_executor);
  return DecodeImageResult{new DecodeImageResult::Impl{
      std::move(task)
          .scheduleOn(detail::get_demux_executor(demux_executor))
          .start()}};
}

BatchDecodeImageResult decoding::async_batch_decode_image(
    const std::vector<std::string>& srcs,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  if (srcs.size() == 0) {
    SPDL_FAIL("At least one image source must be provided.");
  }
  return BatchDecodeImageResult{new BatchDecodeImageResult::Impl{
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
