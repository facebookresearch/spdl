#include <libspdl/core/future.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/future.h"

#include <folly/experimental/coro/Sleep.h>
#include <folly/logging/xlog.h>

#include <chrono>

namespace spdl::core {

FuturePtr async_sleep(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    int milliseconds,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::Task<int> {
    XLOG(INFO) << "CHECKING THE CANCEL POINT";
    co_await folly::coro::co_safe_point;
    XLOG(INFO) << "SLEEP";
    co_await folly::coro::sleep(std::chrono::milliseconds(milliseconds));
    XLOG(INFO) << "WOKE UP";
    co_await folly::coro::co_safe_point;
    XLOG(INFO) << "THROWING";
    throw std::runtime_error("async_sleep was not cancelled.");
  });

  return detail::execute_task_with_callback<int>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

// Function for test
FuturePtr async_sleep_multi(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    int milliseconds,
    int count,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::AsyncGenerator<int> {
    for (int i = 0; i < count; i++) {
      XLOG(INFO) << "CHECKING THE CANCEL POINT: " << i;
      co_await folly::coro::co_safe_point;
      XLOG(INFO) << "SLEEP: " << i;
      co_await folly::coro::sleep(std::chrono::milliseconds(milliseconds));
      XLOG(INFO) << "YIELD: " << i;
      co_yield i;
    }
    XLOG(INFO) << "THROWING ";
    throw std::runtime_error("async_sleep_multi was not cancelled.");
  });

  return detail::execute_generator_with_callback<int>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

} // namespace spdl::core
