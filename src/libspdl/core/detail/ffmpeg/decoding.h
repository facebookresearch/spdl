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
    std::string_view src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdoptorPtr adoptor,
    const IOConfig io_cfg);

// Demuxer for a single image
folly::coro::Task<ImagePacketsPtr> demux_image(
    std::string_view src,
    SourceAdoptorPtr adoptor,
    const IOConfig io_cfg);

// Apply bitstream filter for NVDEC video decoding
folly::coro::Task<VideoPacketsPtr> apply_bsf(VideoPacketsPtr packets);

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

} // namespace spdl::core::detail
