#pragma once
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <optional>

namespace spdl::core::detail {

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    std::optional<AVRational> frame_rate = std::nullopt);

std::string get_video_filter_description(
    const std::optional<Rational>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt);

std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt);

// for debug
std::string describe_graph(AVFilterGraph* graph);

} // namespace spdl::core::detail
