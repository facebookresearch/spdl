#include <libspdl/core/logging.h>

#include <fmt/core.h>

namespace spdl::core::detail {

// TODO: Add stacktrace
std::string get_err_str(
    const std::string_view msg,
    const source_location& location) {
  return fmt::format("{} ({}:{})", msg, location.file_name(), location.line());
}

std::string get_internal_err_str(
    const std::string_view msg,
    const source_location& location) {
  return fmt::format(
      "[INTERNAL ASSERTION FAILED] {} ({}:{})",
      msg,
      location.file_name(),
      location.line());
}

} // namespace spdl::core::detail
