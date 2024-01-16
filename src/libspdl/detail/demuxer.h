#pragma once

#include <libspdl/detail/packets.h>
#include <libspdl/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>

#include <string>
#include <tuple>
#include <vector>

namespace spdl::detail {

folly::coro::AsyncGenerator<PackagedAVPackets&&> stream_demux(
    const enum AVMediaType type,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const IOConfig& cfg);

} // namespace spdl::detail
