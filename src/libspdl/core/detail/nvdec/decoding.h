#pragma once

#include <libspdl/core/detail/ffmpeg/package.h>

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core::detail {

// Apply some bit-stream-filtering for certain video formats
folly::coro::AsyncGenerator<std::unique_ptr<PackagedAVPackets>>
stream_demux_nvdec(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

folly::coro::Task<std::unique_ptr<DecodedFrames>> decode_packets_nvdec(
    std::unique_ptr<PackagedAVPackets> packets,
    int cuda_device_index,
    int crop_left = 0,
    int crop_top = 0,
    int crop_right = 0,
    int crop_bottom = 0,
    int target_width = -1,
    int target_height = -1);

} // namespace spdl::core::detail
