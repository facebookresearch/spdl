#pragma once
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <folly/experimental/coro/AsyncGenerator.h>

#include <optional>

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

folly::coro::AsyncGenerator<AVFramePtr&&> filter_frame(
    FilterGraph& filter_graph,
    AVFramePtr&& frame);

} // namespace spdl::core::detail
