#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core::detail {

CPUBufferPtr convert_audio_frames(const FFmpegAudioFrames* frames);

CPUBufferPtr convert_video_frames_cpu(const std::vector<AVFrame*>& frames);

CUDABufferPtr convert_video_frames_cuda(
    const std::vector<AVFrame*>& frames,
    int device_index);

} // namespace spdl::core::detail
