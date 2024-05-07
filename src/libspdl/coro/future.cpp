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

} // namespace spdl::coro
