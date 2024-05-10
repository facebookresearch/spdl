#pragma once

#include <libspdl/coro/future.h>

#include <folly/Executor.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

#include <memory>

namespace spdl::coro {

struct Future::Impl {
  folly::SemiFuture<folly::Unit> future;
  folly::CancellationSource cs;
  Impl(folly::SemiFuture<folly::Unit>&&, folly::CancellationSource&&);
};

namespace detail {

template <typename ValueType>
folly::coro::Task<void> run_task_with_callbacks(
    folly::coro::Task<ValueType>&& task,
    std::function<void(ValueType)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    folly::CancellationToken&& token) {
  try {
    set_result(
        co_await folly::coro::co_withCancellation(token, std::move(task)));
    co_return;
  } catch (folly::OperationCancelled& e) {
    // Do not log cancellation exceptions.
    // It is good to know that the cancellation working,
    // but the message is same, and it floods the log
    // when tasks are bulk-cancelled.
    XLOG(DBG5) << e.what();
    notify_exception(e.what(), true);
    throw;
  } catch (std::exception& e) {
    XLOG(DBG5) << e.what();
    notify_exception(e.what(), false);
    throw;
  } catch (...) {
    XLOG(CRITICAL) << "Unexpected exception was caught.";
    notify_exception("Unexpected exception.", false);
    throw;
  }
}

template <typename ValueType>
folly::coro::Task<void> run_generator_with_callbacks(
    folly::coro::AsyncGenerator<ValueType>&& gen,
    std::function<void(ValueType)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    folly::CancellationToken&& token) {
  try {
    while (auto val =
               co_await folly::coro::co_withCancellation(token, gen.next())) {
      set_result(std::move(*val));
    }
    co_return;
  } catch (folly::OperationCancelled& e) {
    // Do not log cancellation exceptions.
    // It is good to know that the cancellation working,
    // but the message is same, and it floods the log
    // when tasks are bulk-cancelled.
    XLOG(DBG5) << e.what();
    notify_exception(e.what(), true);
    throw;
  } catch (std::exception& e) {
    XLOG(DBG5) << e.what();
    notify_exception(e.what(), false);
    throw;
  } catch (...) {
    XLOG(CRITICAL) << "Unexpected exception was caught.";
    notify_exception("Unexpected exception.", false);
    throw;
  }
}

FuturePtr run_in_executor(
    folly::coro::Task<void>&& task,
    folly::Executor::KeepAlive<> executor,
    folly::CancellationSource&& cs);

template <typename ValueType>
FuturePtr execute_task_with_callback(
    folly::coro::Task<ValueType>&& task,
    std::function<void(ValueType)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  auto cs = folly::CancellationSource{};
  auto task_cb = folly::coro::co_invoke(
      &run_task_with_callbacks<ValueType>,
      std::move(task),
      set_result,
      notify_exception,
      cs.getToken());
  return run_in_executor(std::move(task_cb), executor, std::move(cs));
}

template <typename ValueType>
FuturePtr execute_generator_with_callback(
    folly::coro::AsyncGenerator<ValueType>&& generator,
    std::function<void(ValueType)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  auto cs = folly::CancellationSource{};
  auto task_cb = folly::coro::co_invoke(
      &run_generator_with_callbacks<ValueType>,
      std::move(generator),
      set_result,
      notify_exception,
      cs.getToken());
  return run_in_executor(std::move(task_cb), executor, std::move(cs));
}

} // namespace detail
} // namespace spdl::coro
