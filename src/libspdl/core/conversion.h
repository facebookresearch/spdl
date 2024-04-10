#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/future.h>

#include <functional>
#include <optional>
#include <vector>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Conversion functions
////////////////////////////////////////////////////////////////////////////////
BufferPtr convert_audio_frames(
    const FFmpegAudioFramesWrapperPtr frames,
    const std::optional<int>& index = std::nullopt);

// FFmpeg video/image could be on CUDA
template <MediaType media_type, bool cpu_only>
BufferPtr convert_vision_frames(
    const FFmpegFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index =
        std::nullopt) requires(media_type != MediaType::Audio);

template <bool cpu_only>
BufferPtr convert_batch_image_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch_frames,
    const std::optional<int>& index = std::nullopt);

template <MediaType media_type>
CUDABuffer2DPitchPtr convert_nvdec_frames(
    const NvDecFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index = std::nullopt);

CUDABuffer2DPitchPtr convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFramesWrapperPtr>& batch_frames,
    const std::optional<int>& index = std::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Async wrapper - FFmpeg
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, bool cpu_only = false>
FuturePtr async_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

template <bool cpu_only>
FuturePtr async_batch_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    const std::optional<int>& index = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

////////////////////////////////////////////////////////////////////////////////
// Async wrapper NVDEC
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    NvDecFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

FuturePtr async_batch_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<NvDecImageFramesWrapperPtr> frames,
    const std::optional<int>& index = std::nullopt,
    ThreadPoolExecutorPtr demux_executor = nullptr);

} // namespace spdl::core
