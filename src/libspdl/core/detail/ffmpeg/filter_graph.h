#pragma once
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

class FilterGraph;

// Helper structure that converts filtering operation (while loop) into
// iterator.
struct IterativeFiltering {
  struct Sentinel {};

  static constexpr Sentinel sentinel{};

  FilterGraph* filter_graph;
  AVFrame* frame;
  bool flush_null;

  struct Ite {
    FilterGraph* filter_graph;
    bool completed = false;
    bool null_flushed;
    AVFramePtr next_ret{};

    bool operator!=(const Sentinel&);

    Ite(FilterGraph*, AVFrame* frame, bool flush_null);

    Ite& operator++();

    AVFramePtr operator*();

   private:
    void fill_next();
  };

  IterativeFiltering(FilterGraph*, AVFrame*, bool flush_null = false);

  Ite begin();
  const Sentinel& end();
};

// Wrap AVFilterGraphPtr to provide convenient methods
class FilterGraph {
  AVFilterGraphPtr graph;

 public:
  FilterGraph(AVFilterGraphPtr&& g) : graph(std::move(g)) {}
  FilterGraph(const AVFilterGraphPtr&) = delete;
  FilterGraph& operator=(const AVFilterGraphPtr&) = delete;

  IterativeFiltering filter(AVFrame*, bool flush_null = false);

  Rational get_src_time_base() const;
  Rational get_sink_time_base() const;

 private:
  void add_frame(AVFrame* in_frame);
  int get_frame(AVFrame* out_frame);

  friend class IterativeFiltering;
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
