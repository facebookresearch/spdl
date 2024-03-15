#pragma once

#include <utility>

namespace spdl::core {

// Simple wrapper around
// `folly::SemiFuture<folly::Unit>` (task without return)
struct Future {
  struct Impl;

 private:
  Impl* pimpl;

 public:
  Future(Impl* p = nullptr);

  Future(const Future&) = delete;
  Future& operator=(const Future&) = delete;
  Future(Future&&) noexcept;
  Future& operator=(Future&&) noexcept;
  ~Future();

  /// Rethrow the internal error.
  /// Does nothing if there was no internal error.
  /// If the future is not complete, it will throw `FutureNotReady` exception.
  void rethrow();
};

} // namespace spdl::core
