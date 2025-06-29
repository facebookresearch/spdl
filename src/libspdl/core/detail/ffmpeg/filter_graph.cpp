/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/filter_graph.h"

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <cmath>
#include <cstring>
#include <ranges>
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
// FilterGraphImpl
////////////////////////////////////////////////////////////////////////////////
namespace {
AVFilterGraphPtr make_graph(const std::string& filter_desc) {
  AVFilterGraphPtr graph{CHECK_AVALLOCATE(avfilter_graph_alloc())};
  graph->nb_threads = 1;

  AVFilterInOut* ins = nullptr;
  AVFilterInOut* outs = nullptr;

  CHECK_AVERROR(
      avfilter_graph_parse2(graph.get(), filter_desc.c_str(), &ins, &outs),
      fmt::format("Failed to parse the filter description: {}", filter_desc))
  CHECK_AVERROR(
      avfilter_graph_config(graph.get(), nullptr),
      fmt::format("Failed to configure the filter graph: `{}`", filter_desc))

  avfilter_inout_free(&ins);
  avfilter_inout_free(&outs);

  return graph;
}
} // namespace

FilterGraphImpl::FilterGraphImpl(const std::string& filter_desc)
    : filter_graph(make_graph(filter_desc)) {
  auto* g = filter_graph.get();

  for (int i = 0; i < (int)g->nb_filters; ++i) {
    auto* ctx = g->filters[i];
    if (std::strcmp(ctx->filter->name, "buffer") == 0 ||
        std::strcmp(ctx->filter->name, "abuffer") == 0) {
      inputs.insert({ctx->name, ctx});
    } else if (
        std::strcmp(ctx->filter->name, "buffersink") == 0 ||
        std::strcmp(ctx->filter->name, "abuffersink") == 0) {
      outputs.insert({ctx->name, ctx});
    }
  }
}

namespace {

// Note:
// Without `const`, it does not compile on clang 17.0.6 && fmt 10.1.1
//
// fmt/core.h:1824:16: note: candidate function [with Context =
// fmt::basic_format_context<fmt::appender, char>, T = <std::string>] not
// viable: expects an lvalue for 1st argument
const std::string parse_unmatched(const AVFilterLink* l, const AVFrame* f) {
  std::vector<std::string> parts;
  switch (l->type) {
    case AVMEDIA_TYPE_VIDEO: {
      if (l->format != f->format) {
        parts.emplace_back(fmt::format(
            "pix_fmt ({} != {})",
            av_get_pix_fmt_name((AVPixelFormat)l->format),
            av_get_pix_fmt_name((AVPixelFormat)f->format)));
      }
      if (l->w != f->width || l->h != f->height) {
        parts.emplace_back(fmt::format(
            "video_size ({}x{} != {}x{})", l->w, l->h, f->width, f->height));
      }
      return fmt::format(
          "The following arguments do not match: {}", fmt::join(parts, ", "));
    }
    case AVMEDIA_TYPE_AUDIO: {
      if (l->format != f->format) {
        parts.emplace_back(fmt::format(
            "sample_fmt ({} != {})",
            av_get_sample_fmt_name((AVSampleFormat)l->format),
            av_get_sample_fmt_name((AVSampleFormat)f->format)));
      }
      if (l->sample_rate != f->sample_rate) {
        parts.emplace_back(fmt::format(
            "sample_rate ({} != {})", l->sample_rate, f->sample_rate));
      }
      if (GET_NUM_CHANNELS(l) != GET_NUM_CHANNELS(f)) {
        parts.emplace_back(fmt::format(
            "num_channels ({} != {})",
            GET_NUM_CHANNELS(l),
            GET_NUM_CHANNELS(f)));
      }
      if (GET_LAYOUT(l) != GET_LAYOUT(f)) {
        parts.emplace_back(fmt::format(
            "channel_layout ({} != {})",
            GET_CHANNEL_LAYOUT_STRING(l),
            GET_CHANNEL_LAYOUT_STRING(f)));
      }
      return fmt::format(
          "The following arguments do not match: {}", fmt::join(parts, ", "));
    }
    default:;
  }
  return fmt::format(
      "Unsupported media type ({}).", av_get_media_type_string(l->type));
}

void add_frame(AVFilterContext* filter_ctx, AVFrame* frame) {
  // We use AV_BUFFERSRC_FLAG_PUSH because the frame we use might be a
  // reference frame to a tensor object, in which case, the data might be
  // garbage-collected after this method returns.
  int flags = AV_BUFFERSRC_FLAG_KEEP_REF | AV_BUFFERSRC_FLAG_PUSH;
  TRACE_EVENT("decoding", "av_buffersrc_add_frame_flags");
  // Starting FFmpeg 7, pushing frame to filter graph can return EOF.
  int ret = av_buffersrc_add_frame_flags(filter_ctx, frame, flags);
  if (ret < 0 && ret != AVERROR_EOF) {
    CHECK_AVERROR(
        ret,
        "Failed to pass a frame to filter. {}",
        parse_unmatched(filter_ctx->outputs[0], frame))
  }
}

int get_frame(AVFilterContext* filter_ctx, AVFrame* frame) {
  int ret;
  {
    TRACE_EVENT("decoding", "av_buffersink_get_frame");
    ret = av_buffersink_get_frame(filter_ctx, frame);
  }
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to filter a frame.")
  }
  return ret;
}

} // namespace

void FilterGraphImpl::add_frames(
    const std::string& name,
    const std::vector<AVFrame*>& frames) {
  auto* filter_ctx = inputs.at(name);
  // Note: Not sure if it's okay to push many frames without pulling. Test.
  for (auto* frame : frames) {
    add_frame(filter_ctx, frame);
  }
}

void FilterGraphImpl::add_frames(const std::vector<AVFrame*>& frames) {
  if (inputs.size() != 1) {
    SPDL_FAIL(fmt::format(
        "Key must be provided when there are multiple inputs. "
        "Available values are {}",
        fmt::join(std::views::keys(inputs), ", ")));
  }
  auto* filter_ctx = inputs.cbegin()->second;
  for (auto* frame : frames) {
    add_frame(filter_ctx, frame);
  }
}

void FilterGraphImpl::flush() {
  for (auto const& pair : inputs) {
    AVFilterContext* filter_ctx = pair.second;
    CHECK_AVERROR(
        av_buffersrc_add_frame_flags(
            filter_ctx, nullptr, AV_BUFFERSRC_FLAG_KEEP_REF),
        fmt::format("Failed to flush the pad: {}.", pair.first))
  }
}

std::string FilterGraphImpl::dump() const {
  char* d = avfilter_graph_dump(filter_graph.get(), nullptr);
  std::string repr{d};
  av_free(d);
  return repr;
}

template <MediaType media_type>
FramesPtr<media_type> FilterGraphImpl::get_frames(AVFilterContext* filter_ctx) {
  auto ret =
      std::make_unique<Frames<media_type>>(0, filter_ctx->inputs[0]->time_base);
  while (true) {
    auto frame = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};
    int err = get_frame(filter_ctx, frame.get());
    if (err == AVERROR_EOF || err == AVERROR(EAGAIN)) {
      break;
    }
    ret->push_back(frame.release());
  }
  return ret;
}

AnyFrames FilterGraphImpl::get_frames(const std::string& name) {
  auto* filter_ctx = outputs.at(name);
  switch (filter_ctx->inputs[0]->type) {
    case AVMEDIA_TYPE_AUDIO:
      return get_frames<MediaType::Audio>(filter_ctx);
    case AVMEDIA_TYPE_VIDEO:
      return get_frames<MediaType::Video>(filter_ctx);
    default:;
  }
  SPDL_FAIL(fmt::format(
      "Unsupported output type: {}",
      av_get_media_type_string(filter_ctx->inputs[0]->type)));
}

AnyFrames FilterGraphImpl::get_frames() {
  if (outputs.size() != 1) {
    SPDL_FAIL(fmt::format(
        "Key must be provided when there are multiple inputs. "
        "Available values are {}",
        fmt::join(std::views::keys(outputs), ", ")));
  }
  return this->get_frames(outputs.cbegin()->first);
}

Rational FilterGraphImpl::get_src_time_base() const {
  if (inputs.size() != 1) {
    SPDL_FAIL(
        "get_src_time_base cannot be used when there are multiple outputs.");
  }
  return inputs.cbegin()->second->outputs[0]->time_base;
}

Rational FilterGraphImpl::get_sink_time_base() const {
  if (outputs.size() != 1) {
    SPDL_FAIL(
        "get_sink_time_base cannot be used when there are multiple outputs.");
  }
  return outputs.cbegin()->second->inputs[0]->time_base;
}

Generator<AVFramePtr> FilterGraphImpl::filter(AVFrame* frame) {
#define TS(OBJ, BASE) (static_cast<double>(OBJ->pts) * BASE.num / BASE.den)

  VLOG(9)
      << (frame ? fmt::format(
                      "{:21s} {:.3f} ({})",
                      " --- raw frame:",
                      TS(frame, get_src_time_base()),
                      frame->pts)
                : fmt::format(" --- flush filter graph"));

  add_frame(inputs.cbegin()->second, frame);

  int errnum;
  do {
    AVFramePtr ret = AVFramePtr{CHECK_AVALLOCATE(av_frame_alloc())};
    switch ((errnum = get_frame(outputs.cbegin()->second, ret.get()))) {
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
#undef TS
}

} // namespace spdl::core::detail
