#include <libspdl/core/future.h>

#include "libspdl/core/detail/future.h"

namespace spdl::core {

Future::Impl::Impl(folly::SemiFuture<folly::Unit>&& fut)
    : future(std::move(fut)) {}

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

void Future::rethrow() {
  // Note: If the future is not complete, the
  // `hasException` method throws `FutureNotReady`.
  if (pimpl->future.hasException()) {
    pimpl->future.value();
  }
}

} // namespace spdl::core
