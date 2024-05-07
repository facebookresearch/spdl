#pragma once

#include <libspdl/coro/executor.h>
#include <libspdl/coro/future.h>

#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>

#include <functional>
#include <optional>
#include <vector>

namespace spdl::coro {

using spdl::core::BufferPtr;
using spdl::core::cuda_allocator;
using spdl::core::FFmpegFramesPtr;
using spdl::core::MediaType;

template <MediaType media_type>
FuturePtr async_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<media_type> frames,
    const std::optional<int>& cuda_device_index = std::nullopt,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

FuturePtr async_batch_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegFramesPtr<MediaType::Image>>&& frames,
    const std::optional<int>& cuda_device_index = std::nullopt,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

} // namespace spdl::coro
