#pragma once

#include <libspdl/core/detail/ffmpeg/package.h>

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core::detail {

struct PackagedAVPackets;

folly::coro::AsyncGenerator<std::unique_ptr<PackagedAVPackets>> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_packets(
    std::unique_ptr<PackagedAVPackets> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

} // namespace spdl::core::detail
