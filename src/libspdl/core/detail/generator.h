#include <coroutine>
#include <cstdint>
#include <exception>

namespace spdl::core::detail {

template <typename T>
struct Generator {
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    T value;
    std::exception_ptr exception;
    Generator get_return_object() {
      return Generator(handle_type::from_promise(*this));
    }
    std::suspend_always initial_suspend() {
      return {};
    }
    std::suspend_always final_suspend() noexcept {
      return {};
    }
    void unhandled_exception() {
      exception = std::current_exception();
    }
    template <std::convertible_to<T> From>
    std::suspend_always yield_value(From&& from) {
      value = std::forward<From>(from);
      return {};
    }
    void return_void() {}
  };

  handle_type h_;

  Generator(handle_type h) : h_(h) {}
  ~Generator() {
    h_.destroy();
  }
  explicit operator bool() {
    fill();
    return !h_.done();
  }
  T operator()() {
    fill();
    full = false;
    return std::move(h_.promise().value);
  }

 private:
  bool full = false;

  void fill() {
    if (!full) {
      h_();
      if (h_.promise().exception) {
        std::rethrow_exception(h_.promise().exception);
      }
      full = true;
    }
  }
};
} // namespace spdl::core::detail
