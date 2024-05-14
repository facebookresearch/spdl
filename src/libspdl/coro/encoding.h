#pragma once

#include <libspdl/coro/executor.h>
#include <libspdl/coro/future.h>

#include <libspdl/core/types.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace spdl::coro {
using spdl::core::EncodeConfig;

// Assumption:
// - Uint8
// - CPU
// - CHW or HW
// These need to be assertedon Python side
FuturePtr async_encode_image(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_execption,
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    const std::string& pix_fmt = "rgb24",
    const std::optional<EncodeConfig>& encode_config = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr);

// Assumption:
// - Uint8
// - CUDA
// - CHW or HW
// These need to be assertedon Python side
FuturePtr async_encode_image_cuda(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_execption,
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    const std::string& pix_fmt = "rgb24",
    const std::optional<EncodeConfig>& encode_config = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr);

} // namespace spdl::coro
