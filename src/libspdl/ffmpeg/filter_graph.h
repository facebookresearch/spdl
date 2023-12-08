#pragma once
#include <libspdl/ffmpeg/wrappers.h>
#include <optional>

namespace spdl {

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational time_base);

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational time_base,
    AVRational frame_rate);

std::string get_video_filter_description(
    std::optional<AVRational> frame_rate,
    std::optional<int> width,
    std::optional<int> height,
    std::optional<std::string> format);

// for debug
std::string describe_graph(AVFilterGraph* graph);

} // namespace spdl
