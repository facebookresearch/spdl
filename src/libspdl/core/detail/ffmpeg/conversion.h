#pragma once
#include <libspdl/core/buffers.h>
#include <libspdl/core/frames.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core::detail {

std::unique_ptr<Buffer> convert_audio_frames(
    const FFmpegAudioFrames* frames,
    const std::optional<int>& index);

std::unique_ptr<Buffer> convert_video_frames_cpu(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index);

std::unique_ptr<Buffer> convert_video_frames_cuda(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index);

} // namespace spdl::core::detail
