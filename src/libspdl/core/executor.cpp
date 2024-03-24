#include <libspdl/core/executor.h>

#include "libspdl/core/detail/executor.h"

namespace spdl::core {

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

} // namespace spdl::core
