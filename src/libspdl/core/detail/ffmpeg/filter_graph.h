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

std::vector<AVFramePtr> filter_frame(
    FilterGraph& filter_graph,
    AVFrame* frame,
    bool flush_null = false);

FilterGraph get_image_enc_filter(
    int src_width,
    int src_height,
    AVPixelFormat src_fmt,
    int enc_width,
    int enc_height,
    const std::optional<std::string>& scale_algo,
    AVPixelFormat enc_fmt,
    const std::optional<std::string>& filter_desc);

} // namespace spdl::core::detail
