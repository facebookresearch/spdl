#pragma once

#include <libspdl/core/future.h>

#include <folly/Executor.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>

#include <memory>

namespace spdl::core {

struct Future::Impl {
  folly::SemiFuture<folly::Unit> future;
  Impl(folly::SemiFuture<folly::Unit>&&);
};

namespace detail {

template <typename ValueType>
std::unique_ptr<Future> execute_task_with_callback(
    folly::coro::Task<ValueType>&& task,
    std::function<void(ValueType)> set_result,
    std::function<void()> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  auto task_cb = folly::coro::co_invoke(
      [=](folly::coro::Task<ValueType>&& t) -> folly::coro::Task<void> {
        try {
          set_result(co_await std::move(t));
          co_return;
        } catch (...) {
          notify_exception();
          throw;
        }
      },
      std::move(task));

  auto future = std::move(task_cb).scheduleOn(executor).start();
  return std::make_unique<Future>(new Future::Impl(std::move(future)));
}

template <typename ValueType>
std::unique_ptr<Future> execute_generator_with_callback(
    folly::coro::AsyncGenerator<ValueType>&& generator,
    std::function<void(std::optional<ValueType>)> set_result,
    std::function<void()> notify_exception,
    folly::Executor::KeepAlive<> executor) {
  using folly::coro::AsyncGenerator;
  using folly::coro::Task;

  auto task_cb = folly::coro::co_invoke(
      [=](AsyncGenerator<ValueType>&& gen) -> Task<void> {
        try {
          while (auto val = co_await gen.next()) {
            set_result({*val});
          }
          set_result(std::nullopt);
          co_return;
        } catch (...) {
          notify_exception();
          throw;
        }
      },
      std::move(generator));

  auto future = std::move(task_cb).scheduleOn(executor).start();
  return std::make_unique<Future>(new Future::Impl(std::move(future)));
}

} // namespace detail
} // namespace spdl::core
