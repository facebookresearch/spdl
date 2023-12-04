#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Task.h>

#include <glog/logging.h>

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
    LOG(ERROR) << "Deleting Foo* " << p << std::endl;
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
    LOG(ERROR) << "Deleting Bar: " << foos.size();
    std::for_each(foos.begin(), foos.end(), [](Foo* p) { delete p; });
  }
  Bar(Bar&& other) = default;
};

folly::coro::AsyncGenerator<Bar&&> gen_move2() {
  co_yield Bar(1);
  co_yield Bar(2);
  co_yield Bar(3);
}

folly::coro::Task<void> run_gen() {
  auto g = gen_move2();
  while (auto result = co_await g.next()) {
    LOG(INFO) << "val: " << result->foo->val;
  }
  co_return;
}

int main(int argc, char** argv) {
  folly::coro::blockingWait(
      run_gen().scheduleOn(folly::getGlobalCPUExecutor().get()));
}
