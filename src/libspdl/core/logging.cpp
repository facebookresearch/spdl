#include <libspdl/core/logging.h>

#include <fmt/core.h>

namespace spdl::core::detail {

// TODO: Add stacktrace
std::string get_err_str(
    const std::string_view msg,
    const source_location& location) {
  return fmt::format(
      "{} ({}:{} - {})",
      msg,
      location.file_name(),
      location.line(),
      location.function_name());
}

std::string get_internal_err_str(
    const std::string_view msg,
    const source_location& location) {
  return fmt::format(
      "[INTERNAL FAILURE] {} ({}:{} - {})",
      msg,
      location.file_name(),
      location.line(),
      location.function_name());
}

} // namespace spdl::core::detail
