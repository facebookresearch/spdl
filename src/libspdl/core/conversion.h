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
BufferPtr convert_audio_frames(const FFmpegAudioFramesWrapperPtr frames);

// FFmpeg video/image could be on CUDA
template <MediaType media_type>
BufferPtr convert_vision_frames(const FFmpegFramesWrapperPtr<media_type> frames)
  requires(media_type != MediaType::Audio);

BufferPtr convert_batch_image_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch_frames);

template <MediaType media_type>
CUDABuffer2DPitchPtr convert_nvdec_frames(
    const NvDecFramesWrapperPtr<media_type> frames);

CUDABuffer2DPitchPtr convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFramesWrapperPtr>& batch_frames);

////////////////////////////////////////////////////////////////////////////////
// CPU to CUDA conversion
////////////////////////////////////////////////////////////////////////////////
BufferPtr convert_to_cuda(BufferPtr buffer, int cuda_device_index);

FuturePtr async_convert_to_cuda(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    BufferWrapperPtr buffer,
    int cuda_device_index,
    ThreadPoolExecutorPtr demux_executor = nullptr);

////////////////////////////////////////////////////////////////////////////////
// Async wrapper - FFmpeg
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FuturePtr async_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<media_type> frames,
    ThreadPoolExecutorPtr demux_executor = nullptr);

FuturePtr async_batch_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr demux_executor = nullptr);

////////////////////////////////////////////////////////////////////////////////
// Async wrapper NVDEC
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    NvDecFramesWrapperPtr<media_type> frames,
    ThreadPoolExecutorPtr demux_executor = nullptr);

FuturePtr async_batch_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<NvDecImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr demux_executor = nullptr);

} // namespace spdl::core
