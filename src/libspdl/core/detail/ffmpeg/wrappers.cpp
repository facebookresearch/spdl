#include <libspdl/core/detail/ffmpeg/wrappers.h>

#include <libspdl/core/detail/tracing.h>

namespace spdl::core::detail {

void free_av_io_ctx(AVIOContext* p) {
  TRACE_EVENT("decoding", "AVIOContext::~AVIOContext");
  if (p) {
    avio_flush(p);
    av_freep(&p->buffer);
  }
  av_freep(&p);
}

void AVIOContextDeleter::operator()(AVIOContext* p) {
  free_av_io_ctx(p);
};

void free_av_fmt_ctx(AVFormatContext* p) {
  TRACE_EVENT("decoding", "avformat_close_input");
  avformat_close_input(&p);
}

void AVFormatInputContextDeleter::operator()(AVFormatContext* p) {
  free_av_fmt_ctx(p);
};

void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  // Tracing because it takes as long as 300ms in case of cuvid decoder.
  TRACE_EVENT("decoding", "avcodec_free_context");
  avcodec_free_context(&p);
}

void AVBSFContextDeleter::operator()(AVBSFContext* p) {
  av_bsf_free(&p);
}

void AVPacketDeleter::operator()(AVPacket* p) {
  if (p) {
    av_packet_unref(p);
    av_packet_free(&p);
  }
}

void AVFrameDeleter::operator()(AVFrame* p) {
  if (p) {
    av_frame_unref(p);
    av_frame_free(&p);
  }
}

void AVBufferRefDeleter::operator()(AVBufferRef* p) {
  TRACE_EVENT("decoding", "av_buffer_unref");
  av_buffer_unref(&p);
}

void AVFilterGraphDeleter::operator()(AVFilterGraph* p) {
  avfilter_graph_free(&p);
}

AVFrameAutoUnref::AVFrameAutoUnref(AVFrame* p_) : p(p_){};
AVFrameAutoUnref::~AVFrameAutoUnref() {
  av_frame_unref(p);
}
} // namespace spdl::core::detail

#ifdef SPDL_DEBUG_REFCOUNT

#include <fmt/core.h>
#include <folly/logging/xlog.h>

extern "C" {
#include <stdatomic.h>
struct AVBuffer {
  uint8_t* data;
  size_t size;
  atomic_uint refcount;
  void (*free)(void* opaque, uint8_t* data);
  void* opaque;
  int flags;
  int flags_internal;
};
}

namespace spdl::core::detail {
void debug_log_avframe_refcount(AVFrame* p) {
  for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
    if (!p->data[i]) {
      break;
    }
    auto buf = p->buf[i]->buffer;
    XLOG(DBG) << fmt::format(
        "Refcount {}: {} ({} -> {})",
        i,
        buf->refcount,
        (void*)(buf->data),
        (void*)(p->data[i]));
  }
}

void debug_log_avpacket_refcount(AVPacket* p) {
  auto buf = p->buf->buffer;
  XLOG(INFO) << fmt::format(
      "Refcount: {} ({})", buf->refcount, (void*)(buf->data));
}
} // namespace spdl::core::detail
#endif
