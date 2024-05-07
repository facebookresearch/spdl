#pragma once

#include <memory>
#include <string>

namespace spdl::coro {

struct ThreadPoolExecutor;

using ThreadPoolExecutorPtr = std::shared_ptr<ThreadPoolExecutor>;

// Abstraction of folly Executor.
struct ThreadPoolExecutor {
  struct Impl;

  Impl* impl = nullptr;

  ThreadPoolExecutor(size_t num_threads, const std::string& thread_name_prefix);
  ~ThreadPoolExecutor();

  size_t get_task_queue_size() const;
};

void trace_default_demux_executor_queue_size();
void trace_default_decode_executor_queue_size();

} // namespace spdl::coro
