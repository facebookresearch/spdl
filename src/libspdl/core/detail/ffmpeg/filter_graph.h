#pragma once
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <optional>

namespace spdl::core::detail {

class FilterGraph {
  AVFilterGraphPtr graph;

 public:
  FilterGraph(AVFilterGraphPtr&& g) : graph(std::move(g)) {}
  FilterGraph(const AVFilterGraphPtr&) = delete;
  FilterGraph& operator=(const AVFilterGraphPtr&) = delete;

  AVFilterContext* get_src_ctx();
  AVFilterContext* get_sink_ctx();
  Rational get_time_base() const;
};

FilterGraph get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

FilterGraph get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    Rational frame_rate);

FilterGraph get_image_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

} // namespace spdl::core::detail
