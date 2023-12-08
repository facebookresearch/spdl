#pragma once
#include <libspdl/ffmpeg/wrappers.h>

namespace spdl {

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational time_base,
    AVRational frame_rate);

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational time_base);

} // namespace spdl
