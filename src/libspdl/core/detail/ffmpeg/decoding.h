#pragma once

#include <libspdl/core/packets.h>

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

folly::coro::AsyncGenerator<std::unique_ptr<DemuxedPackets>> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

// Demuxer for a single image
folly::coro::Task<std::unique_ptr<DemuxedPackets>> demux_image(
    const std::string src,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

folly::coro::Task<FramesPtr> decode_packets(
    std::unique_ptr<DemuxedPackets> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

} // namespace spdl::core::detail
