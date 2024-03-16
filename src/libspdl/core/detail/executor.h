#pragma once

#include <libspdl/core/executor.h>

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <memory>

namespace spdl::core {
namespace detail {

folly::Executor::KeepAlive<> get_demux_executor(ThreadPoolExecutorPtr& exe);

folly::Executor::KeepAlive<> get_decode_executor(ThreadPoolExecutorPtr& exe);

} // namespace detail

struct ThreadPoolExecutor::Impl {
  std::shared_ptr<folly::CPUThreadPoolExecutor> exec;

  Impl(
      size_t num_threads,
      const std::string& thread_name_prefix,
      int throttle_interval);

  folly::Executor::KeepAlive<> get();
};

} // namespace spdl::core
