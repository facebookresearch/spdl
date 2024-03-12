#include <libspdl/core/frames.h>
#include <libspdl/core/result.h>

#include <libspdl/core/detail/logging.h>
#include <libspdl/core/detail/result.h>

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
template <typename ResultType, MediaType media_type>
Result<ResultType, media_type>::Impl::Impl(SemiFuture<ResultType>&& f)
    : future(std::move(f)){};

template <typename ResultType, MediaType media_type>
ResultType Result<ResultType, media_type>::Impl::get() {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  return folly::coro::blockingWait(std::move(future));
}

////////////////////////////////////////////////////////////////////////////////
// Result
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType, MediaType media_type>
Result<ResultType, media_type>::Result(Impl* i) : pimpl(i) {}

template <typename ResultType, MediaType media_type>
Result<ResultType, media_type>::Result(Result&& other) noexcept {
  *this = std::move(other);
}

template <typename ResultType, MediaType media_type>
Result<ResultType, media_type>& Result<ResultType, media_type>::operator=(
    Result&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <typename ResultType, MediaType media_type>
Result<ResultType, media_type>::~Result() {
  delete pimpl;
}

template <typename ResultType, MediaType media_type>
ResultType Result<ResultType, media_type>::get() {
  return pimpl->get();
}

////////////////////////////////////////////////////////////////////////////////
// Results::Impl
////////////////////////////////////////////////////////////////////////////////
namespace {
Task<std::vector<FramesPtr>> check(
    SemiFuture<std::vector<SemiFuture<FramesPtr>>>&&
        future,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<FramesPtr> results;
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

Task<std::vector<FramesPtr>> check_image(
    SemiFuture<std::vector<SemiFuture<FramesPtr>>>&&
        future,
    const std::vector<std::string>& srcs,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<FramesPtr> results;
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

template <typename ResultType, MediaType media_type>
Results<ResultType, media_type>::Impl::Impl(
    std::vector<std::string> srcs_,
    std::vector<std::tuple<double, double>> ts_,
    SemiFuture<std::vector<SemiFuture<ResultType>>>&& future_)
    : srcs(std::move(srcs_)),
      timestamps(std::move(ts_)),
      future(std::move(future_)){};

template <typename ResultType, MediaType media_type>
std::vector<ResultType> Results<ResultType, media_type>::Impl::get(
    bool strict) {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  if constexpr (media_type == MediaType::Image) {
    return blockingWait(check_image(std::move(future), srcs, strict));
  } else {
    return blockingWait(check(std::move(future), srcs[0], timestamps, strict));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Results
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType, MediaType media_type>
Results<ResultType, media_type>::Results(Impl* i) : pimpl(i) {}

template <typename ResultType, MediaType media_type>
Results<ResultType, media_type>::Results(Results&& other) noexcept {
  *this = std::move(other);
}

template <typename ResultType, MediaType media_type>
Results<ResultType, media_type>& Results<ResultType, media_type>::operator=(
    Results&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <typename ResultType, MediaType media_type>
Results<ResultType, media_type>::~Results() {
  delete pimpl;
}

template <typename ResultType, MediaType media_type>
std::vector<ResultType> Results<ResultType, media_type>::get(bool strict) {
  return pimpl->get(strict);
}

// Explicit instantiation
template class Result<FramesPtr, MediaType::Image>;

template class Results<FramesPtr, MediaType::Audio>;
template class Results<FramesPtr, MediaType::Video>;
template class Results<FramesPtr, MediaType::Image>;

} // namespace spdl::core
