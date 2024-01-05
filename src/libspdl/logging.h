#include <fmt/core.h>
#include <exception>

#define SPDL_FAIL(msg)      \
  throw std::runtime_error( \
      fmt::format("{} ({}:{} - {})", msg, __FILE__, __LINE__, __func__))

#define SPDL_FAIL_INTERNAL(msg)             \
  throw std::runtime_error(fmt::format(     \
      "[INTERNAL FAILURE] {} ({}:{} - {})", \
      msg,                                  \
      __FILE__,                             \
      __LINE__,                             \
      __func__))
