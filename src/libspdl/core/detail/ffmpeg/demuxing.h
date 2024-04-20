#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>

#include <string_view>
#include <tuple>
#include <vector>

namespace spdl::core::detail {

template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux(
    std::string_view src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor,
    const std::optional<IOConfig> io_cfg);

// Demuxer for a single image
folly::coro::Task<ImagePacketsPtr> demux_image(
    std::string_view src,
    SourceAdaptorPtr adaptor,
    const std::optional<IOConfig> io_cfg);

// Apply bitstream filter for NVDEC video decoding
folly::coro::Task<VideoPacketsPtr> apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core::detail
