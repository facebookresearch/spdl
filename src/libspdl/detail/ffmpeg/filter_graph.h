#pragma once
#include <libspdl/detail/ffmpeg/wrappers.h>
#include <libspdl/types.h>
#include <optional>

namespace spdl::detail {

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational frame_rate);

// If src_fmt is provided, and format and src_fmt matches,
// the format conversion is omitted.
std::string get_video_filter_description(
    const std::optional<Rational> frame_rate,
    const std::optional<int> width,
    const std::optional<int> height,
    const std::optional<std::string> pix_fmt,
    const enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_NONE);

// for debug
std::string describe_graph(AVFilterGraph* graph);

} // namespace spdl::detail
