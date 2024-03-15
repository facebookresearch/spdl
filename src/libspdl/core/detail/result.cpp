#include <libspdl/core/frames.h>
#include <libspdl/core/result.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/result.h"

#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::blockingWait;
using folly::coro::collectAllTryRange;
using folly::coro::Task;

////////////////////////////////////////////////////////////////////////////////
// Result::Impl
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType>
Result<ResultType>::Impl::Impl(SemiFuture<ResultType>&& f)
    : future(std::move(f)){};

template <typename ResultType>
ResultType Result<ResultType>::Impl::get() {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  return folly::coro::blockingWait(std::move(future));
}

////////////////////////////////////////////////////////////////////////////////
// Result
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType>
Result<ResultType>::Result(Impl* i) : pimpl(i) {}

template <typename ResultType>
Result<ResultType>::Result(Result&& other) noexcept {
  *this = std::move(other);
}

template <typename ResultType>
Result<ResultType>& Result<ResultType>::operator=(Result&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <typename ResultType>
Result<ResultType>::~Result() {
  delete pimpl;
}

template <typename ResultType>
ResultType Result<ResultType>::get() {
  return pimpl->get();
}

////////////////////////////////////////////////////////////////////////////////
// Results::Impl
////////////////////////////////////////////////////////////////////////////////
namespace {
template <typename ResultType>
Task<std::vector<ResultType>> check(
    SemiFuture<std::vector<SemiFuture<ResultType>>>&& future,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<ResultType> results;
  int i = -1;
  folly::exception_wrapper e;
  for (auto& result : co_await collectAllTryRange(std::move(futures))) {
    ++i;
    if (result.hasValue()) {
      results.emplace_back(std::move(result.value()));
      continue;
    }
    e = result.exception();
    XLOG(ERR) << fmt::format(
        "Failed to decode {} ({}-{}) {}",
        src,
        std::get<0>(timestamps[i]),
        std::get<1>(timestamps[i]),
        e.get_exception()->what());
  }
  if (results.size() == 0 || (e.type() != nullptr && strict)) {
    e.throw_exception();
  }
  co_return results;
}

template <typename ResultType>
Task<std::vector<ResultType>> check_image(
    SemiFuture<std::vector<SemiFuture<ResultType>>>&& future,
    const std::vector<std::string>& srcs,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<ResultType> results;
  int i = -1;
  folly::exception_wrapper e;
  for (auto& result : co_await collectAllTryRange(std::move(futures))) {
    ++i;
    if (result.hasValue()) {
      results.emplace_back(std::move(result.value()));
      continue;
    }
    e = result.exception();
    XLOG(ERR) << fmt::format(
        "Failed to decode {}. {}", srcs[i], e.get_exception()->what());
  }
  if (results.size() == 0 || (e.type() != nullptr && strict)) {
    e.throw_exception();
  }
  co_return results;
}

} // namespace

template <typename ResultType>
Results<ResultType>::Impl::Impl(
    std::vector<std::string> srcs_,
    std::vector<std::tuple<double, double>> ts_,
    SemiFuture<std::vector<SemiFuture<ResultType>>>&& future_)
    : srcs(std::move(srcs_)),
      timestamps(std::move(ts_)),
      future(std::move(future_)){};

template <typename ResultType>
std::vector<ResultType> Results<ResultType>::Impl::get(bool strict) {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  if constexpr (
      std::is_same<ResultType, std::unique_ptr<FFmpegImageFrames>>::value ||
      std::is_same<ResultType, std::unique_ptr<NvDecImageFrames>>::value) {
    return blockingWait(
        check_image<ResultType>(std::move(future), srcs, strict));
  } else {
    return blockingWait(check(std::move(future), srcs[0], timestamps, strict));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Results
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType>
Results<ResultType>::Results(Impl* i) : pimpl(i) {}

template <typename ResultType>
Results<ResultType>::Results(Results&& other) noexcept {
  *this = std::move(other);
}

template <typename ResultType>
Results<ResultType>& Results<ResultType>::operator=(Results&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <typename ResultType>
Results<ResultType>::~Results() {
  delete pimpl;
}

template <typename ResultType>
std::vector<ResultType> Results<ResultType>::get(bool strict) {
  return pimpl->get(strict);
}

// Explicit instantiation
template class Result<std::unique_ptr<FFmpegImageFrames>>;
template class Result<std::unique_ptr<NvDecImageFrames>>;

template class Results<std::unique_ptr<FFmpegAudioFrames>>;
template class Results<std::unique_ptr<FFmpegVideoFrames>>;
template class Results<std::unique_ptr<FFmpegImageFrames>>;
template class Results<std::unique_ptr<NvDecVideoFrames>>;
template class Results<std::unique_ptr<NvDecImageFrames>>;

} // namespace spdl::core
