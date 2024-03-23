#include <folly/CancellationToken.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Task.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

folly::coro::AsyncGenerator<int> gen() {
  for (int i = 0; i < 5; ++i) {
    co_yield i;
  }
}

struct Foo {
  int val;
  Foo(int val) : val(val) {}
};

struct Deleter {
  void operator()(Foo* p) {
    XLOG(ERR) << "  Deleting Foo* " << p;
    delete p;
  }
};

using FooPtr = std::unique_ptr<Foo, Deleter>;

folly::coro::AsyncGenerator<FooPtr> gen_move() {
  auto foo = FooPtr{new Foo{1}};
  co_yield std::move(foo);
  co_yield FooPtr{new Foo{2}};
  co_yield FooPtr{new Foo{3}};
}

struct Bar {
  FooPtr foo;
  Bar(int val) : foo(new Foo{val}), foos(val) {}
  std::vector<Foo*> foos{};
  ~Bar() {
    XLOG(ERR) << "Deleting Bar: " << foos.size();
    std::for_each(foos.begin(), foos.end(), [](Foo* p) {
      XLOG(ERR) << "Foo* = " << p;
      delete p;
    });
  }
  Bar(Bar&& other) = default;
};

folly::coro::AsyncGenerator<Bar&&> gen_move2() {
  co_yield Bar(1);
  co_yield Bar(2);
  co_await folly::coro::co_safe_point;
  co_yield Bar(3);
}

folly::coro::Task<void> run_gen() {
  auto g = gen_move2();
  while (auto result = co_await g.next()) {
    XLOG(INFO) << "val: " << result->foo->val;
  }
  co_return;
}

template <typename T>
folly::coro::Task<T> run_task(folly::coro::Task<T>&& task) {
  auto cs = folly::CancellationSource{};

  auto result = co_await folly::coro::co_invoke(
      [&](folly::CancellationSource cs_, folly::coro::Task<T>&& task_)
          -> folly::coro::Task<folly::Try<folly::Unit>> {
        cs_.requestCancellation();
        co_return co_await folly::coro::co_awaitTry(
            folly::coro::co_withCancellation(cs_.getToken(), std::move(task_)));
      },
      cs,
      std::move(task));
  if (const std::exception* ex = result.tryGetExceptionObject()) {
    LOG(ERROR) << "Failed with error: " << ex->what();
  }
}

int main(int nargs, char** argv) {
  {
    folly::Init init{&nargs, &argv};
    folly::coro::blockingWait(run_task<void>(run_gen()).scheduleOn(
        folly::getGlobalCPUExecutor().get()));
  }
}
