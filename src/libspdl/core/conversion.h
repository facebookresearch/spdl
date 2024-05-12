#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>

#include <vector>

namespace spdl::core {

CPUBufferPtr convert_audio_frames(const FFmpegAudioFrames* frames);

CPUBufferPtr convert_video_frames(const std::vector<AVFrame*>& frames);

} // namespace spdl::core
