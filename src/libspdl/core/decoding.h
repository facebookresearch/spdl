#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/future.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Asynchronous decodings
////////////////////////////////////////////////////////////////////////////////

/// Decode audio, video or image
template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsWrapperPtr<media_type> packets,
    const std::optional<DecodeConfig>& decode_config,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

/// Decode video or image
template <MediaType media_type>
FuturePtr async_decode_nvdec(
    std::function<void(NvDecFramesWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsWrapperPtr<media_type> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr decode_executor);

// Function for test
FuturePtr async_sleep(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    int milliseconds,
    ThreadPoolExecutorPtr executor);

// Function for test
FuturePtr async_sleep_multi(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    int milliseconds,
    int count,
    ThreadPoolExecutorPtr executor);

} // namespace spdl::core
