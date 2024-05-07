#include <libspdl/coro/executor.h>

#include "libspdl/coro/detail/executor.h"

namespace spdl::coro {

ThreadPoolExecutor::ThreadPoolExecutor(
    size_t num_threads,
    const std::string& thread_name_prefix)
    : impl(new Impl(num_threads, thread_name_prefix)) {}

ThreadPoolExecutor::~ThreadPoolExecutor() {
  delete impl;
}

size_t ThreadPoolExecutor::get_task_queue_size() const {
  return impl->get_task_queue_size();
}

} // namespace spdl::coro
