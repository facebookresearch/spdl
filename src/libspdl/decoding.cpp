#include <libspdl/decoding.h>
#include <libspdl/detail/executors.h>
#include <libspdl/detail/ffmpeg/decoding.h>
#include <libspdl/logging.h>

#include <fmt/core.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <cstddef>
#include <cstdint>

using folly::coro::collectAllTryRange;

namespace spdl {
namespace {
folly::coro::Task<std::vector<FrameContainer>> stream_decode(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::string filter_desc,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg) {
  auto demuxer = detail::stream_demux(type, src, timestamps, io_cfg);
  std::vector<folly::SemiFuture<FrameContainer>> futures;
  auto exec = detail::getDecoderThreadPoolExecutor();
  while (auto packets = co_await demuxer.next()) {
    auto task = detail::decode_packets(*packets, filter_desc, decode_cfg);
    futures.emplace_back(std::move(task).scheduleOn(exec).start());
  }
  XLOG(DBG) << "Waiting for decode jobs to finish";
  std::vector<FrameContainer> results;
  size_t i = 0;
  for (auto& result : co_await collectAllTryRange(std::move(futures))) {
    if (result.hasValue()) {
      results.emplace_back(std::move(result.value()));
    } else {
      XLOG(ERR) << fmt::format(
          "Failed to decode video clip. Error: {} (Source: {}, timestamp: {}, {})",
          result.exception().what(),
          src,
          std::get<0>(timestamps[i]),
          std::get<1>(timestamps[i]));
    }
    ++i;
  };
  if (results.size() != timestamps.size()) {
    SPDL_FAIL("Failed to decode some video clips. Check the error log.");
  }
  co_return results;
}
} // namespace

std::vector<FrameContainer> decode_video(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg) {
  auto job = stream_decode(
      MediaType::Video, src, timestamps, filter_desc, io_cfg, decode_cfg);
  return folly::coro::blockingWait(
      std::move(job).scheduleOn(detail::getDemuxerThreadPoolExecutor()));
}

std::vector<FrameContainer> decode_audio(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg) {
  auto job = stream_decode(
      MediaType::Audio, src, timestamps, filter_desc, io_cfg, decode_cfg);
  return folly::coro::blockingWait(
      std::move(job).scheduleOn(detail::getDemuxerThreadPoolExecutor()));
}

} // namespace spdl
