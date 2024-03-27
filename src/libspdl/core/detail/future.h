#pragma once

#include <libspdl/core/future.h>

#include <folly/Executor.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

#include <memory>

namespace spdl::core {

struct Future::Impl {
  folly::SemiFuture<folly::Unit> future;
  folly::CancellationSource cs;
  Impl(folly::SemiFuture<folly::Unit>&&, folly::CancellationSource&&);
  void cancel();
};

namespace detail {

template <typename ValueType>
FuturePtr execute_task_with_callback(
    folly::coro::Task<ValueType>&& task,
    std::function<void(ValueType)> set_result,
    std::function<void()> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  auto cs = folly::CancellationSource{};
  auto task_cb = folly::coro::co_invoke(
      [=](folly::coro::Task<ValueType>&& task,
          folly::CancellationToken&& token) -> folly::coro::Task<void> {
        try {
          set_result(co_await folly::coro::co_withCancellation(
              token, std::move(task)));
          co_return;
        } catch (folly::OperationCancelled& e) {
          // Do not log cancellation exceptions.
          // It is good to know that the cancellation working,
          // but the message is same, and it floods the log
          // when tasks are bulk-cancelled.
          XLOG(DBG5) << e.what();
          notify_exception();
          throw;
        } catch (std::exception& e) {
          XLOG(ERR) << e.what();
          notify_exception();
          throw;
        } catch (...) {
          notify_exception();
          throw;
        }
      },
      std::move(task),
      cs.getToken());

  auto future = std::move(task_cb).scheduleOn(executor).start();
  return std::make_unique<Future>(
      new Future::Impl(std::move(future), std::move(cs)));
}

template <typename ValueType>
FuturePtr execute_generator_with_callback(
    folly::coro::AsyncGenerator<ValueType>&& generator,
    std::function<void(std::optional<ValueType>)> set_result,
    std::function<void()> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  using folly::coro::AsyncGenerator;
  using folly::coro::Task;

  auto cs = folly::CancellationSource{};
  auto task_cb = folly::coro::co_invoke(
      [=](AsyncGenerator<ValueType>&& gen,
          folly::CancellationToken&& token) -> Task<void> {
        try {
          while (auto val = co_await gen.next()) {
            set_result({*val});
          }
          set_result(std::nullopt);
          co_return;
        } catch (folly::OperationCancelled& e) {
          // Do not log cancellation exceptions.
          // It is good to know that the cancellation working,
          // but the message is same, and it floods the log
          // when tasks are bulk-cancelled.
          XLOG(DBG5) << e.what();
          notify_exception();
          throw;
        } catch (std::exception& e) {
          XLOG(ERR) << e.what();
          notify_exception();
          throw;
        } catch (...) {
          notify_exception();
          throw;
        }
      },
      std::move(generator),
      cs.getToken());

  auto future = std::move(task_cb).scheduleOn(executor).start();
  return std::make_unique<Future>(
      new Future::Impl(std::move(future), std::move(cs)));
}

} // namespace detail
} // namespace spdl::core
