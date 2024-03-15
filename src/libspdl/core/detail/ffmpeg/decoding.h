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

template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

// Demuxer for a single image
folly::coro::Task<PacketsPtr<MediaType::Image>> demux_image(
    const std::string src,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

} // namespace spdl::core::detail
