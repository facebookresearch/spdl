#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core::detail {

BufferPtr convert_audio_frames(
    const FFmpegAudioFrames* frames,
    const std::optional<int>& index);

BufferPtr convert_video_frames_cpu(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index);

BufferPtr convert_video_frames_cuda(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index);

} // namespace spdl::core::detail
