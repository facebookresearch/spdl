/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/filter_graph.h"

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <cmath>
#include <stdexcept>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core::detail {

////////////////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> get_filters() {
  std::vector<std::string> ret;
  void* t = nullptr;
  const AVFilter* filter;
  while (!(filter = av_filter_iterate(&t))) {
    ret.emplace_back(filter->name);
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// FilterGraph
////////////////////////////////////////////////////////////////////////////////

namespace {
// for debug
std::string describe_graph(AVFilterGraph* graph) {
  char* desc_ = avfilter_graph_dump(graph, nullptr);
  std::string desc{desc_};
  av_free(static_cast<void*>(desc_));
  return desc;
}

std::string get_buffer_arg(
    int width,
    int height,
    const char* pix_fmt_name,
    AVRational time_base,
    AVRational sample_aspect_ratio) {
  return fmt::format(
      "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
      width,
      height,
      pix_fmt_name,
      time_base.num,
      time_base.den,
      sample_aspect_ratio.num,
      sample_aspect_ratio.den);
}

std::string get_buffer_arg(
    int width,
    int height,
    const char* pix_fmt_name,
    AVRational time_base,
    Rational frame_rate,
    AVRational sample_aspect_ratio) {
  return fmt::format(
      "video_size={}x{}:pix_fmt={}:time_base={}/{}:frame_rate={}/{}:pixel_aspect={}/{}",
      width,
      height,
      pix_fmt_name,
      time_base.num,
      time_base.den,
      frame_rate.num,
      frame_rate.den,
      sample_aspect_ratio.num,
      sample_aspect_ratio.den);
}

std::string get_abuffer_arg(
    AVRational time_base,
    int sample_rate,
    const char* sample_fmt_name,
    int nb_channels) {
  return fmt::format(
      "time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout={}c",
      time_base.num,
      time_base.den,
      sample_rate,
      sample_fmt_name,
      nb_channels);
}

/// @brief Create a new filter context given its description and options.
/// @param graph The filter graph where the filter will be created.
/// @param flt The type of filter
/// @param name The name to give to the newly-created filter instance.
/// @param args Configuration of the filter instance.
AVFilterContext* create_filter(
    AVFilterGraph* graph,
    const AVFilter* flt,
    const char* name,
    const char* args = nullptr) {
  AVFilterContext* flt_ctx = nullptr;
  TRACE_EVENT("decoding", "avfilter_graph_create_filter");
  CHECK_AVERROR(
      avfilter_graph_create_filter(&flt_ctx, flt, name, args, nullptr, graph),
      "Failed to create input filter: {}({})",
      flt->name,
      args);
  return flt_ctx;
}

DEF_DPtr(AVFilterInOut, avfilter_inout_free); // this defines AVFilterInOutDPtr

AVFilterInOutDPtr get_io(const char* name) {
  AVFilterInOutDPtr io{CHECK_AVALLOCATE(avfilter_inout_alloc())};
  io->name = av_strdup(name);
  io->pad_idx = 0;
  io->next = nullptr;
  return io;
}

AVFilterGraphPtr alloc_filter_graph() {
  TRACE_EVENT("decoding", "avfilter_graph_alloc");
  AVFilterGraph* ptr = CHECK_AVALLOCATE(avfilter_graph_alloc());
  ptr->nb_threads = 1;
  return AVFilterGraphPtr{ptr};
}

FilterGraph get_filter(
    const char* desc,
    const AVFilter* src,
    const char* src_arg,
    const AVFilter* sink) {
  auto filter_graph = alloc_filter_graph();
  auto p = filter_graph.get();

  // 1. Define the filters at the ends
  AVFilterInOutDPtr in = get_io("in");
  AVFilterInOutDPtr out = get_io("out");

  // Note
  // AVFilterContext* are attached to the graph and will be freed when the
  // graph is freed. So we don't need to free them here.
  in->filter_ctx = create_filter(p, src, "in", src_arg);
  out->filter_ctx = create_filter(p, sink, "out");
  {
    TRACE_EVENT("decoding", "avfilter_graph_parse_ptr");
    CHECK_AVERROR(
        avfilter_graph_parse_ptr(p, desc, out, in, nullptr),
        "Failed to create filter from: \"{}\"",
        desc);
  }

  // 3. Create the filter graph
  {
    TRACE_EVENT("decoding", "avfilter_graph_config");
    CHECK_AVERROR(
        avfilter_graph_config(p, nullptr),
        "Failed to configure the graph. \"{}\"",
        desc);
  }

  // for (unsigned i = 0; i < p->nb_filters; ++i) {
  //   LOG(INFO) << "Filter " << i << ": " << p->filters[i]->name;
  // }

  VLOG(5) << describe_graph(filter_graph.get());

  return FilterGraph{std::move(filter_graph)};
}

#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

int add_frame(AVFilterContext* src_ctx, AVFrame* frame) {
  int ret;
  {
    TRACE_EVENT("decoding", "av_buffersrc_add_frame_flags");
    ret = av_buffersrc_add_frame_flags(
        src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
  }
  if (ret < 0 && ret != AVERROR_EOF) {
    CHECK_AVERROR_NUM(ret, "Failed to pass a frame to filter.");
  }
  return ret;
}

int get_frame(AVFilterContext* sink_ctx, AVFrame* frame) {
  int ret;
  {
    TRACE_EVENT("decoding", "av_buffersink_get_frame");
    ret = av_buffersink_get_frame(sink_ctx, frame);
  }
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to filter a frame.");
  }
  return ret;
}

} // namespace

Generator<AVFramePtr> FilterGraph::filter(AVFrame* frame) {
  VLOG(9)
      << (frame ? fmt::format(
                      "{:21s} {:.3f} ({})",
                      " --- raw frame:",
                      TS(frame, get_src_time_base()),
                      frame->pts)
                : fmt::format(" --- flush filter graph"));

  if (add_frame(graph->filters[0], frame) == AVERROR_EOF) {
    co_return;
  }

  int errnum;
  do {
    AVFramePtr ret = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};
    switch ((errnum = get_frame(graph->filters[1], ret.get()))) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_return;
      default: {
        VLOG(9) << fmt::format(
            "{:21s} {:.3f} ({})",
            " ---- filtered frame:",
            TS(ret, get_sink_time_base()),
            ret->pts);

        co_yield std::move(ret);
      }
    }
  } while (errnum >= 0);
}

Rational FilterGraph::get_src_time_base() const {
  auto ctx = graph->filters[0]->outputs[0];
  return Rational{ctx->time_base.num, ctx->time_base.den};
}

Rational FilterGraph::get_sink_time_base() const {
  auto ctx = graph->filters[1]->inputs[0];
  return Rational{ctx->time_base.num, ctx->time_base.den};
}

////////////////////////////////////////////////////////////////////////////////
// Factory functions - decoding
////////////////////////////////////////////////////////////////////////////////

FilterGraph get_audio_filter(
    const std::string& filter_desc,
    AVCodecContext* codec_ctx) {
  if (filter_desc.empty()) {
    SPDL_FAIL("filter description is empty.");
  }
  auto arg = get_abuffer_arg(
      codec_ctx->pkt_timebase,
      codec_ctx->sample_rate,
      av_get_sample_fmt_name(codec_ctx->sample_fmt),
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 2, 100)
      codec_ctx->ch_layout.nb_channels
#else
      codec_ctx->channels
#endif
  );
  return get_filter(
      filter_desc.c_str(),
      avfilter_get_by_name("abuffer"),
      arg.c_str(),
      avfilter_get_by_name("abuffersink"));
}

FilterGraph get_video_filter(
    const std::string& filter_desc,
    AVCodecContext* codec_ctx,
    Rational frame_rate) {
  if (filter_desc.empty()) {
    SPDL_FAIL("filter description is empty.");
  }
  auto arg = get_buffer_arg(
      codec_ctx->width,
      codec_ctx->height,
      av_get_pix_fmt_name(codec_ctx->pix_fmt),
      codec_ctx->pkt_timebase,
      frame_rate,
      codec_ctx->sample_aspect_ratio);
  return get_filter(
      filter_desc.c_str(),
      avfilter_get_by_name("buffer"),
      arg.c_str(),
      avfilter_get_by_name("buffersink"));
}

FilterGraph get_image_filter(
    const std::string& filter_desc,
    AVCodecContext* codec_ctx) {
  if (filter_desc.empty()) {
    SPDL_FAIL("filter description is empty.");
  }
  auto arg = get_buffer_arg(
      codec_ctx->width,
      codec_ctx->height,
      av_get_pix_fmt_name(codec_ctx->pix_fmt),
      codec_ctx->pkt_timebase,
      codec_ctx->sample_aspect_ratio);
  return get_filter(
      filter_desc.c_str(),
      avfilter_get_by_name("buffer"),
      arg.c_str(),
      avfilter_get_by_name("buffersink"));
}

////////////////////////////////////////////////////////////////////////////////
// Factory functions - encoding
////////////////////////////////////////////////////////////////////////////////

FilterGraph get_image_enc_filter(
    int src_width,
    int src_height,
    AVPixelFormat src_fmt,
    int enc_width,
    int enc_height,
    const std::optional<std::string>& scale_algo,
    AVPixelFormat enc_fmt,
    const std::optional<std::string>& filter_desc) {
  auto desc = [&]() -> std::string {
    std::vector<std::string> parts;
    if (filter_desc) {
      parts.emplace_back(filter_desc.value());
    }
    if (filter_desc || src_width != enc_width || src_height != enc_height) {
      std::string arg = fmt::format("scale={}:{}", enc_width, enc_height);
      if (scale_algo) {
        arg += fmt::format(":flags={}", scale_algo.value());
      }
      parts.emplace_back(arg);
    }
    if (filter_desc || src_fmt != enc_fmt) {
      parts.emplace_back(
          fmt::format("format={}", av_get_pix_fmt_name(enc_fmt)));
    }
    if (parts.size()) {
      return fmt::to_string(fmt::join(parts, ","));
    }
    return "null";
  }();
  auto arg = get_buffer_arg(
      src_width, src_height, av_get_pix_fmt_name(src_fmt), {1, 1}, {1, 1});
  return get_filter(
      desc.c_str(),
      avfilter_get_by_name("buffer"),
      arg.c_str(),
      avfilter_get_by_name("buffersink"));
}

} // namespace spdl::core::detail
