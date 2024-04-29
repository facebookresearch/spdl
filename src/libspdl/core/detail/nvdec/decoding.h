#pragma once

#include <libspdl/core/packets.h>

#include <libspdl/core/adaptor.h>
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

template <MediaType media_type>
folly::coro::Task<NvDecFramesPtr<media_type>> decode_nvdec(
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string> pix_fmt = std::nullopt);

folly::coro::Task<NvDecVideoFramesPtr> decode_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string> pix_fmt = std::nullopt);

} // namespace spdl::core::detail
