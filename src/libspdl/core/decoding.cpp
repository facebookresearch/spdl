#include <libspdl/core/decoding.h>

#include <libspdl/core/detail/executors.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/logging.h>

#include <fmt/core.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <cstddef>
#include <cstdint>

using folly::coro::collectAllTryRange;

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// DecodingResult
////////////////////////////////////////////////////////////////////////////////
struct DecodingResultFuture::Impl {
  bool fetched{false};
  folly::SemiFuture<ResultType> future;

  Impl(folly::SemiFuture<ResultType>&& fut) : future(std::move(fut)){};

  ResultType get() {
    if (fetched) {
      SPDL_FAIL("The decoding result is already fetched.");
    }
    fetched = true;
    return folly::coro::blockingWait(std::move(future));
  }
};

////////////////////////////////////////////////////////////////////////////////
// DecodingResult
////////////////////////////////////////////////////////////////////////////////
DecodingResultFuture::DecodingResultFuture(Impl* i) : pimpl(i) {}

DecodingResultFuture::DecodingResultFuture(
    DecodingResultFuture&& other) noexcept {
  *this = std::move(other);
}

DecodingResultFuture& DecodingResultFuture::operator=(
    DecodingResultFuture&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

DecodingResultFuture::~DecodingResultFuture() {
  delete pimpl;
}

auto DecodingResultFuture::get() -> DecodingResultFuture::ResultType {
  return pimpl->get();
}

////////////////////////////////////////////////////////////////////////////////
// Decoding functions
////////////////////////////////////////////////////////////////////////////////
namespace {
folly::coro::Task<std::vector<std::unique_ptr<FrameContainer>>> stream_decode(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc) {
  auto demuxer = detail::stream_demux(type, src, timestamps, adoptor, io_cfg);
  std::vector<folly::SemiFuture<std::unique_ptr<FrameContainer>>> futures;
  auto exec = detail::getDecoderThreadPoolExecutor();
  while (auto packets = co_await demuxer.next()) {
    auto task = detail::decode_packets(*packets, filter_desc, decode_cfg);
    futures.emplace_back(std::move(task).scheduleOn(exec).start());
  }
  XLOG(DBG) << "Waiting for decode jobs to finish";
  std::vector<std::unique_ptr<FrameContainer>> results;
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

DecodingResultFuture decode_video(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc) {
  auto job = stream_decode(
      MediaType::Video,
      src,
      timestamps,
      adoptor,
      io_cfg,
      decode_cfg,
      filter_desc);
  return DecodingResultFuture{new DecodingResultFuture::Impl{
      std::move(job)
          .scheduleOn(detail::getDemuxerThreadPoolExecutor())
          .start()}};
}

DecodingResultFuture decode_audio(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc) {
  auto job = stream_decode(
      MediaType::Audio,
      src,
      timestamps,
      adoptor,
      io_cfg,
      decode_cfg,
      filter_desc);
  return DecodingResultFuture{new DecodingResultFuture::Impl{
      std::move(job)
          .scheduleOn(detail::getDemuxerThreadPoolExecutor())
          .start()}};
}

} // namespace spdl::core
