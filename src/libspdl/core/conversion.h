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
// CPU to CUDA conversion
////////////////////////////////////////////////////////////////////////////////
FuturePtr async_convert_to_cuda(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    BufferWrapperPtr buffer,
    int cuda_device_index,
    ThreadPoolExecutorPtr demux_executor = nullptr);

////////////////////////////////////////////////////////////////////////////////
// FFmpeg
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
// NVDEC
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
