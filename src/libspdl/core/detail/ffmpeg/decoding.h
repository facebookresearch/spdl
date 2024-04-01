#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>

#include <string>

namespace spdl::core::detail {

template <MediaType media_type>
folly::coro::Task<FFmpegFramesPtr<media_type>> decode_packets_ffmpeg(
    PacketsPtr<media_type> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

} // namespace spdl::core::detail
