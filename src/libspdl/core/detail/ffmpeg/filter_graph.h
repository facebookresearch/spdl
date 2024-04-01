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
    std::optional<Rational> frame_rate = std::nullopt);

// for debug
std::string describe_graph(AVFilterGraph* graph);

} // namespace spdl::core::detail
