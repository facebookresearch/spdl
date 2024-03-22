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
template <MediaType media_type, template <MediaType> typename ResultType>
Result<media_type, ResultType>::Impl::Impl(
    SemiFuture<ResultType<media_type>>&& f)
    : future(std::move(f)){};

template <MediaType media_type, template <MediaType> typename ResultType>
ResultType<media_type> Result<media_type, ResultType>::Impl::get() {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  return folly::coro::blockingWait(std::move(future));
}

////////////////////////////////////////////////////////////////////////////////
// Result
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, template <MediaType> typename ResultType>
Result<media_type, ResultType>::Result(Impl* i) : pimpl(i) {}

template <MediaType media_type, template <MediaType> typename ResultType>
Result<media_type, ResultType>::Result(Result&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media_type, template <MediaType> typename ResultType>
Result<media_type, ResultType>& Result<media_type, ResultType>::operator=(
    Result&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <MediaType media_type, template <MediaType> typename ResultType>
Result<media_type, ResultType>::~Result() {
  delete pimpl;
}

template <MediaType media_type, template <MediaType> typename ResultType>
ResultType<media_type> Result<media_type, ResultType>::get() {
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

template <template <MediaType> typename ResultType>
Task<std::vector<ResultType<MediaType::Image>>> check_image(
    SemiFuture<std::vector<SemiFuture<ResultType<MediaType::Image>>>>&& future,
    const std::vector<std::string>& srcs,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<ResultType<MediaType::Image>> results;
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

template <MediaType media_type, template <MediaType> typename ResultType>
Results<media_type, ResultType>::Impl::Impl(
    std::vector<std::string> srcs_,
    std::vector<std::tuple<double, double>> ts_,
    SemiFuture<std::vector<SemiFuture<ResultType<media_type>>>>&& future_)
    : srcs(std::move(srcs_)),
      timestamps(std::move(ts_)),
      future(std::move(future_)){};

template <MediaType media_type, template <MediaType> typename ResultType>
std::vector<ResultType<media_type>> Results<media_type, ResultType>::Impl::get(
    bool strict) {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  if constexpr (media_type == MediaType::Image) {
    return blockingWait(
        check_image<ResultType>(std::move(future), srcs, strict));
  } else {
    return blockingWait(
        check(std::move(future), srcs.at(0), timestamps, strict));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Results
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, template <MediaType> typename ResultType>
Results<media_type, ResultType>::Results(Impl* i) : pimpl(i) {}

template <MediaType media_type, template <MediaType> typename ResultType>
Results<media_type, ResultType>::Results(Results&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media_type, template <MediaType> typename ResultType>
Results<media_type, ResultType>& Results<media_type, ResultType>::operator=(
    Results&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

template <MediaType media_type, template <MediaType> typename ResultType>
Results<media_type, ResultType>::~Results() {
  delete pimpl;
}

template <MediaType media_type, template <MediaType> typename ResultType>
std::vector<ResultType<media_type>> Results<media_type, ResultType>::get(
    bool strict) {
  return pimpl->get(strict);
}

// Explicit instantiation
template class Result<MediaType::Image, FFmpegFramesWrapperPtr>;
template class Result<MediaType::Image, NvDecFramesWrapperPtr>;

template class Results<MediaType::Audio, FFmpegFramesWrapperPtr>;
template class Results<MediaType::Video, FFmpegFramesWrapperPtr>;
template class Results<MediaType::Image, FFmpegFramesWrapperPtr>;
template class Results<MediaType::Video, NvDecFramesWrapperPtr>;
template class Results<MediaType::Image, NvDecFramesWrapperPtr>;

} // namespace spdl::core
