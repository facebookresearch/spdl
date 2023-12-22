#include <fmt/format.h>

#include <libspdl/detail/ffmpeg/filter_graph.h>
#include <libspdl/detail/ffmpeg/logging.h>
#include <stdexcept>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::detail {
namespace {

AVFilterGraphPtr alloc_filter_graph() {
  AVFilterGraph* ptr = CHECK_AVALLOCATE(avfilter_graph_alloc());
  ptr->nb_threads = 1;
  return AVFilterGraphPtr{ptr};
}

std::string get_buffer_arg(
    int width,
    int height,
    const char* pix_fmt_name,
    AVRational time_base,
    AVRational frame_rate,
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
    uint64_t channel_layout) {
  return fmt::format(
      "time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout={:x}",
      time_base.num,
      time_base.den,
      sample_rate,
      sample_fmt_name,
      channel_layout);
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
  CHECK_AVERROR(
      avfilter_graph_create_filter(&flt_ctx, flt, name, args, nullptr, graph),
      "Failed to create input filter: {}({})",
      flt->name,
      args);
  return flt_ctx;
}

DEF_DPtr(AVFilterInOut, avfilter_inout_free); // this defines AVFilterInOutDPtr

AVFilterInOutDPtr get_io(const char* name, AVFilterContext* filter_ctx) {
  AVFilterInOutDPtr io{CHECK_AVALLOCATE(avfilter_inout_alloc())};
  io->name = av_strdup(name);
  io->filter_ctx = filter_ctx;
  io->pad_idx = 0;
  io->next = nullptr;
  return io;
}

AVFilterGraphPtr get_filter(
    const char* desc,
    const AVFilter* src,
    const char* src_arg,
    const AVFilter* sink,
    AVBufferRef* hw_frames_ctx = nullptr) {
  auto filter_graph = alloc_filter_graph();
  auto p = filter_graph.get();

  // 1. Define the filters at the ends
  // Note
  // AVFilterContext* are attached to the graph and will be freed when the
  // graph is freed. So we don't need to free them here.
  auto src_ctx = create_filter(p, src, "in", src_arg);
  auto sink_ctx = create_filter(p, sink, "out");

  // 2. Define the middle
  AVFilterInOutDPtr in = get_io("in", src_ctx);
  AVFilterInOutDPtr out = get_io("out", sink_ctx);
  CHECK_AVERROR(
      avfilter_graph_parse_ptr(p, desc, out, in, nullptr),
      "Failed to create filter from: \"{}\"",
      desc);

  if (hw_frames_ctx) {
    src_ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
  }

  // 3. Create the filter graph
  CHECK_AVERROR(
      avfilter_graph_config(p, nullptr), "Failed to configure the graph.");

  // for (unsigned i = 0; i < p->nb_filters; ++i) {
  //   XLOG(INFO) << "Filter " << i << ": " << p->filters[i]->name;
  // }

  return filter_graph;
}

} // namespace

AVFilterGraphPtr get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx) {
  static const AVFilter* src = avfilter_get_by_name("abuffer");
  static const AVFilter* sink = avfilter_get_by_name("abuffersink");
  auto arg = get_abuffer_arg(
      codec_ctx->pkt_timebase,
      codec_ctx->sample_rate,
      av_get_sample_fmt_name(codec_ctx->sample_fmt),
      codec_ctx->channel_layout);
  return get_filter(filter_description.c_str(), src, arg.c_str(), sink);
}

AVFilterGraphPtr get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    AVRational frame_rate) {
  static const AVFilter* src = avfilter_get_by_name("buffer");
  static const AVFilter* sink = avfilter_get_by_name("buffersink");
  auto arg = get_buffer_arg(
      codec_ctx->width,
      codec_ctx->height,
      av_get_pix_fmt_name(codec_ctx->pix_fmt),
      codec_ctx->pkt_timebase,
      frame_rate,
      codec_ctx->sample_aspect_ratio);
  return get_filter(
      filter_description.c_str(),
      src,
      arg.c_str(),
      sink,
      codec_ctx->hw_frames_ctx);
}

std::string get_video_filter_description(
    const std::optional<Rational> frame_rate,
    const std::optional<int> width,
    const std::optional<int> height,
    const std::optional<std::string> pix_fmt,
    const enum AVPixelFormat src_pix_fmt) {
  std::vector<std::string> parts;
  if (frame_rate) {
    auto fr = frame_rate.value();
    parts.emplace_back(
        fmt::format("fps={}/{}", std::get<0>(fr), std::get<1>(fr)));
  }
  if (width || height) {
    std::vector<std::string> scale;
    if (width) {
      scale.emplace_back(fmt::format("width={}", width.value()));
    }
    if (height > 0) {
      scale.emplace_back(fmt::format("height={}", height.value()));
    }
    parts.push_back(fmt::format("scale={}", fmt::join(scale, ":")));
  }
  if (pix_fmt) {
    auto val = pix_fmt.value();
    if (src_pix_fmt == AV_PIX_FMT_NONE ||
        val != av_get_pix_fmt_name(src_pix_fmt)) {
      parts.push_back(fmt::format("format=pix_fmts={}", val));
    }
  }
  return fmt::to_string(fmt::join(parts, ","));
}

std::string describe_graph(AVFilterGraph* graph) {
  char* desc_ = avfilter_graph_dump(graph, NULL);
  std::string desc{desc_};
  av_free(static_cast<void*>(desc_));
  return desc;
}

} // namespace spdl::detail
