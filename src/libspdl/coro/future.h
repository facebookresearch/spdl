#pragma once

#include <memory>
#include <utility>

namespace spdl::coro {

struct Future;

using FuturePtr = std::unique_ptr<Future>;

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

  bool cancelled() const;

  void cancel();
};

} // namespace spdl::coro
