#pragma once
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

class FilterGraph {
  AVFilterGraphPtr graph;

 public:
  FilterGraph(AVFilterGraphPtr&& g) : graph(std::move(g)) {}
  FilterGraph(const AVFilterGraphPtr&) = delete;
  FilterGraph& operator=(const AVFilterGraphPtr&) = delete;

  void add_frame(AVFrame* in_frame);
  int get_frame(AVFrame* out_frame);

  Rational get_src_time_base() const;
  Rational get_sink_time_base() const;
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

std::vector<AVFramePtr> filter_frame(FilterGraph& filter_graph, AVFrame* frame);

} // namespace spdl::core::detail
