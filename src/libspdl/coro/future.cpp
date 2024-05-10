#include <libspdl/coro/future.h>

#include "libspdl/coro/detail/future.h"

namespace spdl::coro {

Future::Impl::Impl(
    folly::SemiFuture<folly::Unit>&& fut,
    folly::CancellationSource&& cs_)
    : future(std::move(fut)), cs(std::move(cs_)) {}

Future::Future(Future::Impl* p) : pimpl(p) {}

Future::Future(Future&& other) noexcept {
  *this = std::move(other);
}

Future& Future::operator=(Future&& other) noexcept {
  using std::swap;
  swap(pimpl, other.pimpl);
  return *this;
}

Future::~Future() {
  delete pimpl;
}

bool Future::cancelled() const {
  return pimpl->cs.isCancellationRequested();
}

void Future::cancel() {
  pimpl->cs.requestCancellation();
}

namespace detail {

FuturePtr run_in_executor(
    folly::coro::Task<void>&& task,
    folly::Executor::KeepAlive<> executor,
    folly::CancellationSource&& cs) {
  return std::make_unique<Future>(new Future::Impl(
      std::move(task).scheduleOn(executor).start(), std::move(cs)));
}
}

} // namespace spdl::coro
