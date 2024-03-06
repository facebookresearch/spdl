#include <libspdl/core/decoding.h>

#include <libspdl/core/decoding/results.h>
#include <libspdl/core/logging.h>

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

using Output = std::unique_ptr<DecodedFrames>;

////////////////////////////////////////////////////////////////////////////////
// SingleDecodingResult::Impl
////////////////////////////////////////////////////////////////////////////////
SingleDecodingResult::Impl::Impl(SemiFuture<Output>&& f)
    : future(std::move(f)){};

Output SingleDecodingResult::Impl::get() {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  return folly::coro::blockingWait(std::move(future));
}

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

Output SingleDecodingResult::get() {
  return pimpl->get();
}

////////////////////////////////////////////////////////////////////////////////
// MultipleDecodingResult::Impl
////////////////////////////////////////////////////////////////////////////////
namespace {
Task<std::vector<Output>> check(
    SemiFuture<std::vector<SemiFuture<Output>>>&& future,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<Output> results;
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

Task<std::vector<Output>> check_image(
    SemiFuture<std::vector<SemiFuture<Output>>>&& future,
    const std::vector<std::string>& srcs,
    bool strict) {
  auto futures = co_await std::move(future);

  std::vector<Output> results;
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

MultipleDecodingResult::Impl::Impl(
    const enum MediaType type_,
    std::vector<std::string> srcs_,
    std::vector<std::tuple<double, double>> ts_,
    SemiFuture<std::vector<SemiFuture<Output>>>&& future_)
    : type(type_),
      srcs(std::move(srcs_)),
      timestamps(std::move(ts_)),
      future(std::move(future_)){};

std::vector<Output> MultipleDecodingResult::Impl::get(bool strict) {
  if (fetched) {
    SPDL_FAIL("The decoding result is already fetched.");
  }
  fetched = true;
  if (type == MediaType::Image) {
    return blockingWait(check_image(std::move(future), srcs, strict));
  }
  return blockingWait(check(std::move(future), srcs[0], timestamps, strict));
}

////////////////////////////////////////////////////////////////////////////////
// MultipleDecodingResult
////////////////////////////////////////////////////////////////////////////////
MultipleDecodingResult::MultipleDecodingResult(Impl* i) : pimpl(i) {}

MultipleDecodingResult::MultipleDecodingResult(
    MultipleDecodingResult&& other) noexcept {
  *this = std::move(other);
}

MultipleDecodingResult& MultipleDecodingResult::operator=(
    MultipleDecodingResult&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

MultipleDecodingResult::~MultipleDecodingResult() {
  delete pimpl;
}

std::vector<Output> MultipleDecodingResult::get(bool strict) {
  return pimpl->get(strict);
}

} // namespace spdl::core
