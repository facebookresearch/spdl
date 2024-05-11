#pragma once

#include <libspdl/core/packets.h>

#include <libspdl/core/adaptor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core::detail {

template <MediaType media_type>
CUDABufferPtr decode_nvdec(
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string> pix_fmt = std::nullopt,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt);

CUDABufferPtr decode_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string> pix_fmt = std::nullopt,
    bool strict = true,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt);

} // namespace spdl::core::detail
