#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/frames.h>

namespace spdl::core {

std::unique_ptr<Buffer> convert_audio_frames(
    const FFmpegAudioFrames& frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_video_frames(
    const FFmpegVideoFrames& frames,
    const std::optional<int>& index = std::nullopt);

std::unique_ptr<Buffer> convert_image_frames(
    const FFmpegImageFrames& frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl::core
