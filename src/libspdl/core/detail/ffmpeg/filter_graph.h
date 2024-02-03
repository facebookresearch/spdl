#pragma once
#include <libspdl/core/detail/ffmpeg/wrappers.h>
#include <libspdl/core/types.h>
#include <optional>

namespace spdl::detail {

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational frame_rate);

std::string get_video_filter_description(
    const std::optional<Rational>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt);

std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt);

MediaType get_output_media_type(const AVFilterGraph* p);

// for debug
std::string describe_graph(AVFilterGraph* graph);

} // namespace spdl::detail
