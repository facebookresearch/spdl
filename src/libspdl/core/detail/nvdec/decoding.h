#pragma once

#include <libspdl/core/packets.h>

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
template <MediaType media_type>
folly::coro::AsyncGenerator<PacketsPtr<media_type>> stream_demux_nvdec(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    SourceAdoptorPtr adoptor,
    const IOConfig io_cfg);

template <MediaType media_type>
folly::coro::Task<NvDecFramesPtr<media_type>> decode_nvdec(
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string> pix_fmt = std::nullopt,
    bool is_image = false);

} // namespace spdl::core::detail
