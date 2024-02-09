#include <libspdl/core/decoding.h>

#include <libspdl/core/detail/executors.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/tracing.h>
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
// DecodingResultFuture::Impl
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
// DecodingResultFuture
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
// async_decode wrapper
////////////////////////////////////////////////////////////////////////////////
using ResultType = std::unique_ptr<FrameContainer>;

namespace {
folly::coro::Task<std::vector<ResultType>> stream_decode_task(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc);
} // namespace

DecodingResultFuture async_decode(
    const enum MediaType type,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc) {
  auto task = stream_decode_task(
      type, src, timestamps, adoptor, io_cfg, decode_cfg, filter_desc);
  return DecodingResultFuture{new DecodingResultFuture::Impl{
      std::move(task)
          .scheduleOn(detail::getDemuxerThreadPoolExecutor())
          .start()}};
}

////////////////////////////////////////////////////////////////////////////////
// Actual implementation of decoding task;
////////////////////////////////////////////////////////////////////////////////
namespace {
folly::coro::Task<std::vector<ResultType>> stream_decode_task(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc) {
  std::vector<folly::SemiFuture<ResultType>> futures;
  {
    auto exec = detail::getDecoderThreadPoolExecutor();
    auto demuxer = detail::stream_demux(
        type, src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = detail::decode_packets(
          *std::move(result), std::move(decode_cfg), std::move(filter_desc));
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }

  XLOG(DBG) << "Waiting for decode jobs to finish";
  std::vector<ResultType> results;
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

} // namespace spdl::core
