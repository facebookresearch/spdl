#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/frames.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Device-specific conversion functions (will fail if wrong device)
////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<Buffer> convert_video_frames_to_cpu_buffer(
    const FFmpegVideoFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_image_frames_to_cpu_buffer(
    const FFmpegImageFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_batch_image_frames_to_cpu_buffer(
    const std::vector<FFmpegImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

// -----------------------------------------------------------------------------

std::unique_ptr<Buffer> convert_video_frames_to_cuda_buffer(
    const FFmpegVideoFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_image_frames_to_cuda_buffer(
    const FFmpegImageFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_batch_image_frames_to_cuda_buffer(
    const std::vector<FFmpegImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Device-agnostic conversion functions (device is picked accordingly
////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<Buffer> convert_audio_frames(
    const FFmpegAudioFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_video_frames(
    const FFmpegVideoFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_image_frames(
    const FFmpegImageFrames* frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_batch_image_frames(
    const std::vector<FFmpegImageFrames*>& batch_frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl::core
