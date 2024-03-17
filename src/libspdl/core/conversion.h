#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/future.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Device-specific conversion functions (will fail if wrong device)
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
CPUBufferPtr convert_visual_frames_to_cpu_buffer(
    const FFmpegFrames<media_type>* frames,
    const std::optional<int>& index = std::nullopt);

CPUBufferPtr convert_batch_image_frames_to_cpu_buffer(
    const std::vector<FFmpegImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Device-agnostic conversion functions (device is picked accordingly
////////////////////////////////////////////////////////////////////////////////
CPUBufferPtr convert_audio_frames(
    const FFmpegAudioFrames* frames,
    const std::optional<int>& index = std::nullopt);

// FFmpeg video/image could be on CUDA
template <MediaType media_type>
BufferPtr convert_visual_frames(
    const FFmpegFrames<media_type>* frames,
    const std::optional<int>& index =
        std::nullopt) requires(media_type != MediaType::Audio);

BufferPtr convert_batch_image_frames(
    const std::vector<FFmpegImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

template <MediaType media_type>
std::shared_ptr<CUDABuffer2DPitch> convert_nvdec_frames(
    const NvDecFrames<media_type>* frames,
    const std::optional<int>& index = std::nullopt);

std::shared_ptr<CUDABuffer2DPitch> convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl::core
