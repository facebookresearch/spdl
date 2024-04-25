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

// Demux audio/video stream from resource indicator
template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux(
    const std::string uri,
    const SourceAdaptorPtr adaptor,
    const std::optional<IOConfig> io_cfg,
    const std::vector<std::tuple<double, double>> timestamps);

// Demux audio/video stream from memory
// _zero_clear sets all the data to zero. This is only for testing.
template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux(
    const std::string_view data,
    const std::optional<IOConfig> io_cfg,
    const std::vector<std::tuple<double, double>> timestamps,
    bool _zero_clear = false);

// Demux a single image from the resource indicator
folly::coro::Task<ImagePacketsPtr> demux_image(
    const std::string uri,
    const SourceAdaptorPtr adaptor,
    const std::optional<IOConfig> io_cfg);

// Demux a single image from the memory
// _zero_clear sets all the data to zero. This is only for testing.
folly::coro::Task<ImagePacketsPtr> demux_image(
    const std::string_view src,
    const std::optional<IOConfig> io_cfg,
    bool _zero_clear = false);

// Apply bitstream filter for NVDEC video decoding
folly::coro::Task<VideoPacketsPtr> apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core::detail
