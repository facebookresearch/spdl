#include <libspdl/core/executor.h>

#include "libspdl/core/detail/executor.h"

namespace spdl::core {

ThreadPoolExecutor::ThreadPoolExecutor(
    size_t num_threads,
    const std::string& thread_name_prefix,
    int throttle_interval)
    : impl(new Impl(num_threads, thread_name_prefix, throttle_interval)) {}

ThreadPoolExecutor::~ThreadPoolExecutor() {
  delete impl;
}

} // namespace spdl::core
