#include <libspdl/core/future.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/future.h"

#include <folly/experimental/coro/Sleep.h>

#include <chrono>

namespace spdl::core {

FuturePtr async_sleep(
    std::function<void(int)> set_result,
    std::function<void()> notify_exception,
    int milliseconds,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::Task<int> {
    co_await folly::coro::sleep(std::chrono::milliseconds(milliseconds));
    co_await folly::coro::co_safe_point;
    co_return 1;
  });

  return detail::execute_task_with_callback<int>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

} // namespace spdl::core
