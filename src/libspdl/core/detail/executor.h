#pragma once

#include <folly/Executor.h>

namespace spdl::core::detail {

folly::Executor::KeepAlive<> get_default_demux_executor();
folly::Executor::KeepAlive<> get_default_decode_executor();

} // namespace spdl::core::detail
