#pragma once

#include <libspdl/coro/executor.h>
#include <libspdl/coro/future.h>

#include <libspdl/core/adaptor.h>
#include <libspdl/core/decoding.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace spdl::coro {

using spdl::core::CPUBufferPtr;
using spdl::core::CropArea;
using spdl::core::cuda_allocator;
using spdl::core::CUDABufferPtr;
using spdl::core::DecodeConfig;
using spdl::core::DemuxConfig;
using spdl::core::FFmpegFramesPtr;
using spdl::core::MediaType;
using spdl::core::PacketsPtr;
using spdl::core::SourceAdaptorPtr;

template <MediaType media_type>
using StreamingDecoderPtr =
    std::shared_ptr<spdl::core::StreamingDecoder<media_type>>;

////////////////////////////////////////////////////////////////////////////////
// Decode only
////////////////////////////////////////////////////////////////////////////////

/// Decode media packets
template <MediaType media_type>
FuturePtr async_decode(
    std::function<void(FFmpegFramesPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig> decode_config,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template <MediaType media_type>
FuturePtr async_streaming_decoder(
    std::function<void(StreamingDecoderPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& dmx_cfg,
    const std::string& filter_desc,
    ThreadPoolExecutorPtr executor);

template <MediaType media_type>
FuturePtr async_decode_frames(
    std::function<void(std::optional<FFmpegFramesPtr<media_type>>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    StreamingDecoderPtr<media_type> decoder,
    int num_frames,
    ThreadPoolExecutorPtr executor);

/// Decode video or image
template <MediaType media_type>
FuturePtr async_decode_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt,
    ThreadPoolExecutorPtr decode_executor = nullptr);

FuturePtr async_batch_decode_image_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<PacketsPtr<MediaType::Image>>&& packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict,
    const uintptr_t cuda_stream = 0,
    const std::optional<cuda_allocator>& cuda_allocator = std::nullopt,
    ThreadPoolExecutorPtr decode_executor = nullptr);

////////////////////////////////////////////////////////////////////////////////
// Demux + decode in one step
////////////////////////////////////////////////////////////////////////////////

/// Decode media from source
template <MediaType media_type>
FuturePtr async_decode_from_source(
    std::function<void(FFmpegFramesPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<DecodeConfig>& decode_config,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

template <MediaType media_type>
FuturePtr async_decode_from_bytes(
    std::function<void(FFmpegFramesPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<DecodeConfig>& decode_config,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor,
    bool _zero_clear = false);

////////////////////////////////////////////////////////////////////////////////

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

} // namespace spdl::coro
