#pragma once

#include <folly/Executor.h>

namespace spdl::core::detail {

folly::Executor::KeepAlive<> getDemuxerThreadPoolExecutor();
folly::Executor::KeepAlive<> getDecoderThreadPoolExecutor();

} // namespace spdl::core::detail
