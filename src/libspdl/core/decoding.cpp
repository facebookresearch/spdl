#include <libspdl/core/decoding.h>

#include <libspdl/core/detail/executors.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#ifdef SPDL_USE_NVDEC
#include <libspdl/core/detail/nvdec/decoding.h>
#endif

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
// SingleDecodingResult::Impl
////////////////////////////////////////////////////////////////////////////////
struct SingleDecodingResult::Impl {
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
// SingleDecodingResult
////////////////////////////////////////////////////////////////////////////////
SingleDecodingResult::SingleDecodingResult(Impl* i) : pimpl(i) {}

SingleDecodingResult::SingleDecodingResult(
    SingleDecodingResult&& other) noexcept {
  *this = std::move(other);
}

SingleDecodingResult& SingleDecodingResult::operator=(
    SingleDecodingResult&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

SingleDecodingResult::~SingleDecodingResult() {
  delete pimpl;
}

auto SingleDecodingResult::get() -> SingleDecodingResult::ResultType {
  return pimpl->get();
}

////////////////////////////////////////////////////////////////////////////////
// Actual implementation of decoding task;
////////////////////////////////////////////////////////////////////////////////
using ResultType = std::unique_ptr<DecodedFrames>;
namespace {
folly::coro::Task<std::vector<ResultType>> check_futures(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    std::vector<folly::SemiFuture<ResultType>> futures) {
  std::vector<ResultType> results;
  size_t i = 0;
  for (auto& result : co_await collectAllTryRange(std::move(futures))) {
    if (result.hasValue()) {
      results.emplace_back(std::move(result.value()));
    } else {
      XLOG(ERR) << fmt::format(
          "Failed to decode video clip. (Source: {} {}-{}) Error: {})",
          src,
          std::get<0>(timestamps[i]),
          std::get<1>(timestamps[i]),
          result.exception().what());
    }
    ++i;
  };
  if (results.size() != timestamps.size()) {
    SPDL_FAIL("Failed to decode some video clips. Check the error log.");
  }
  co_return results;
}

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
  co_return co_await check_futures(src, timestamps, std::move(futures));
}

folly::coro::Task<ResultType> image_decode_task(
    const std::string src,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc) {
  auto exec = detail::getDecoderThreadPoolExecutor();
  auto packet =
      co_await detail::demux_image(src, std::move(adoptor), std::move(io_cfg));
  auto task = detail::decode_packets(
      std::move(packet), std::move(decode_cfg), std::move(filter_desc));
  folly::SemiFuture<ResultType> future =
      std::move(task).scheduleOn(exec).start();
  co_return co_await std::move(future);
}

#ifdef SPDL_USE_NVDEC
folly::coro::Task<std::vector<ResultType>> stream_decode_task_nvdec(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int width,
    int height) {
  std::vector<folly::SemiFuture<ResultType>> futures;
  {
    auto exec = detail::getDecoderThreadPoolExecutor();
    auto demuxer = detail::stream_demux_nvdec(
        src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = detail::decode_packets_nvdec(
          *std::move(result),
          cuda_device_index,
          crop_left,
          crop_top,
          crop_right,
          crop_bottom,
          width,
          height);
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return co_await check_futures(src, timestamps, std::move(futures));
}
#endif
} // namespace

////////////////////////////////////////////////////////////////////////////////
// async_decode wrapper
////////////////////////////////////////////////////////////////////////////////
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

SingleDecodingResult async_decode_image(
    const std::string& src,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc) {
  auto task = image_decode_task(src, adoptor, io_cfg, decode_cfg, filter_desc);
  return SingleDecodingResult{new SingleDecodingResult::Impl{
      std::move(task)
          .scheduleOn(detail::getDemuxerThreadPoolExecutor())
          .start()}};
}

namespace {
void validate_nvdec_params(
    int cuda_device_index,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int width,
    int height) {
  if (cuda_device_index < 0) {
    SPDL_FAIL(fmt::format(
        "cuda_device_index must be non-negative. Found: {}",
        cuda_device_index));
  }
  if (crop_left < 0) {
    SPDL_FAIL(
        fmt::format("crop_left must be non-negative. Found: {}", crop_left));
  }
  if (crop_top < 0) {
    SPDL_FAIL(
        fmt::format("crop_top must be non-negative. Found: {}", crop_top));
  }
  if (crop_right < 0) {
    SPDL_FAIL(
        fmt::format("crop_right must be non-negative. Found: {}", crop_right));
  }
  if (crop_bottom < 0) {
    SPDL_FAIL(fmt::format(
        "crop_bottom must be non-negative. Found: {}", crop_bottom));
  }
  if (width > 0 && width % 2) {
    SPDL_FAIL(fmt::format("width must be positive and even. Found: {}", width));
  }
  if (height > 0 && height % 2) {
    SPDL_FAIL(
        fmt::format("height must be positive and even. Found: {}", height));
  }
}

void init_cuda() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TRACE_EVENT("nvdec", "cuInit");
    cuInit(0);
  });
}

} // namespace

DecodingResultFuture async_decode_nvdec(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int width,
    int height) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  validate_nvdec_params(
      cuda_device_index,
      crop_left,
      crop_top,
      crop_right,
      crop_bottom,
      width,
      height);
  init_cuda();

  auto task = stream_decode_task_nvdec(
      src,
      timestamps,
      cuda_device_index,
      adoptor,
      io_cfg,
      crop_left,
      crop_top,
      crop_right,
      crop_bottom,
      width,
      height);
  return DecodingResultFuture{new DecodingResultFuture::Impl{
      std::move(task)
          .scheduleOn(detail::getDemuxerThreadPoolExecutor())
          .start()}};
#endif
}

} // namespace spdl::core
