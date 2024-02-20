#pragma once

#include <exception>
#include <string_view>
#include <version>

#if defined __cpp_lib_source_location
#include <source_location>
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#else
#error \
    "Neither <source_location> or <experimental/source_location> is available."
#endif

namespace spdl::core::detail {

#if defined __cpp_lib_source_location
using std::source_location;
#else
using std::experimental::source_location;
#endif

std::string get_err_str(
    const std::string_view msg,
    const source_location& location);

std::string get_internal_err_str(
    const std::string_view msg,
    const source_location& location);

} // namespace spdl::core::detail

#define SPDL_FAIL(msg)                                      \
  throw std::runtime_error(spdl::core::detail::get_err_str( \
      msg, spdl::core::detail::source_location::current()))

#define SPDL_FAIL_INTERNAL(msg)                                      \
  throw std::runtime_error(spdl::core::detail::get_internal_err_str( \
      msg, spdl::core::detail::source_location::current()))
