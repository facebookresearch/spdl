include(CheckCXXSourceCompiles)

function(check_coroutine_support)

  check_cxx_source_compiles("
      #include <coroutine>

      struct Task {
        struct promise_type {
          void return_void() {}
          std::suspend_never initial_suspend() { return {}; }
          std::suspend_always final_suspend() noexcept { return {}; }
          Task get_return_object() { return{}; }
          void unhandled_exception() {}
        };
      };

      Task foo() {
        co_return;
      }

      int main() {}"

    HAS_COROUTINE)

  if (NOT HAS_COROUTINE)
     message(FATAL_ERROR "C++ compiler does not support coroutine.")
  endif()
endfunction()

check_coroutine_support()
