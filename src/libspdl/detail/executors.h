#pragma once

#include <folly/Executor.h>

namespace spdl::detail {

folly::Executor::KeepAlive<> getDemuxerThreadPoolExecutor();
folly::Executor::KeepAlive<> getDecoderThreadPoolExecutor();

} // namespace spdl::detail
