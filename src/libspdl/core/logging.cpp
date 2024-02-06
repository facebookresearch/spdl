#include <libspdl/core/logging.h>

#include <fmt/core.h>

namespace spdl::core::detail {

// TODO: Add stacktrace

std::string format_err(
    const std::string& msg,
    const std::string& file,
    const int line,
    const std::string& func) {
  return fmt::format("{} ({}:{} - {})", msg, file, line, func);
}

std::string format_err_internal(
    const std::string& msg,
    const std::string& file,
    const int line,
    const std::string& func) {
  return fmt::format(
      "[INTERNAL FAILURE] {} ({}:{} - {})", msg, file, line, func);
}

} // namespace spdl::core::detail
