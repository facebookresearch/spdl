#pragma once

#include <exception>
#include <string>

namespace spdl::core::detail {

std::string format_err(
    const std::string& msg,
    const std::string& file,
    const int line,
    const std::string& func);

std::string format_err_internal(
    const std::string& msg,
    const std::string& file,
    const int line,
    const std::string& func);

} // namespace spdl::core::detail

#define SPDL_FAIL(msg)      \
  throw std::runtime_error( \
      spdl::core::detail::format_err(msg, __FILE__, __LINE__, __func__));

#define SPDL_FAIL_INTERNAL(msg)                                     \
  throw std::runtime_error(spdl::core::detail::format_err_internal( \
      msg, __FILE__, __LINE__, __func__));
