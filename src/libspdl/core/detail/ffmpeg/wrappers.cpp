#include <libspdl/core/detail/ffmpeg/wrappers.h>

namespace spdl::core::detail {

void free_av_io_ctx(AVIOContext* p) {
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
  avformat_close_input(&p);
}

void AVFormatInputContextDeleter::operator()(AVFormatContext* p) {
  free_av_fmt_ctx(p);
};

void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  avcodec_free_context(&p);
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
