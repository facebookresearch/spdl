#pragma once

#include <string>

namespace spdl::core {

// Abstraction of folly Executor.
struct ThreadPoolExecutor {
  struct Impl;

  Impl* impl = nullptr;

  // throttle_interval
  // When enabled, use unbounded blocking queues with ThrottledLifoSem.
  //
  // * <0   : Disable
  // * 0<=  : wake up interval [micro second]
  //
  // See
  // https://github.com/facebook/folly/blob/4aeca3a9fb13bee2c3d4f59fdf529d1d82dfe50a/folly/synchronization/ThrottledLifoSem.h#L35-L69
  ThreadPoolExecutor(
      size_t num_threads,
      const std::string& thread_name_prefix,
      int throttle_interval);
  ~ThreadPoolExecutor();
};

} // namespace spdl::core
