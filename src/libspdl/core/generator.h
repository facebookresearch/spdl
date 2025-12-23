/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <coroutine>
#include <exception>

namespace spdl::core {

/// Coroutine-based generator for lazy iteration.
///
/// Generator provides a C++20 coroutine-based iterator that yields values
/// on demand. Used for streaming operations where producing all values
/// upfront would be inefficient.
///
/// @tparam T Type of values yielded by the generator.
template <typename T>
struct Generator {
  /// Promise type for coroutine support.
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  /// Coroutine promise implementation.
  struct promise_type {
    /// Current yielded value.
    T value;
    /// Exception captured during generation.
    std::exception_ptr exception;

    /// Create generator from promise.
    Generator get_return_object() {
      return Generator(handle_type::from_promise(*this));
    }

    /// Suspend at coroutine start.
    std::suspend_always initial_suspend() {
      return {};
    }

    /// Suspend at coroutine end.
    std::suspend_always final_suspend() noexcept {
      return {};
    }

    /// Handle unhandled exceptions.
    void unhandled_exception() {
      exception = std::current_exception();
    }

    /// Yield a value from the coroutine.
    template <std::convertible_to<T> From>
    std::suspend_always yield_value(From&& from) {
      value = std::forward<From>(from);
      return {};
    }

    /// Complete coroutine without returning a value.
    void return_void() {}
  };

  handle_type h_;

  /// Construct generator from coroutine handle.
  explicit Generator(handle_type h) : h_(h) {}

  /// Deleted copy constructor.
  Generator(const Generator&) = delete;

  /// Deleted copy assignment operator.
  Generator& operator=(const Generator&) = delete;

  /// Move constructor - transfers ownership and nulls source handle.
  Generator(Generator&& other) noexcept : h_(other.h_), full_(other.full_) {
    other.h_ = nullptr;
    other.full_ = false;
  }

  /// Move assignment operator - transfers ownership and nulls source handle.
  Generator& operator=(Generator&& other) noexcept {
    if (this != &other) {
      if (h_) {
        h_.destroy();
      }
      h_ = other.h_;
      full_ = other.full_;
      other.h_ = nullptr;
      other.full_ = false;
    }
    return *this;
  }

  /// Destructor destroys the coroutine handle if valid.
  ~Generator() {
    if (h_) {
      h_.destroy();
    }
  }

  /// Check if more values are available.
  ///
  /// @return true if generator has more values, false otherwise.
  explicit operator bool() {
    fill();
    return !h_.done();
  }

  /// Get the next value from the generator.
  ///
  /// @return Next yielded value.
  T operator()() {
    fill();
    full_ = false;
    return std::move(h_.promise().value);
  }

 private:
  bool full_ = false;

  void fill() {
    if (!full_) {
      h_();
      if (h_.promise().exception) {
        std::rethrow_exception(h_.promise().exception);
      }
      full_ = true;
    }
  }
};
} // namespace spdl::core
