#include <libspdl/ffmpeg/logging.h>
#include <libspdl/ffmpeg/wrappers.h>

namespace spdl {

////////////////////////////////////////////////////////////////////////////////
// RAII wrappers
////////////////////////////////////////////////////////////////////////////////

void AVIOContextDeleter::operator()(AVIOContext* p) {
  if (p) {
    avio_flush(p);
    av_freep(&p->buffer);
  }
  av_freep(&p);
};

void AVFormatInputContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  avcodec_free_context(&p);
}

void AVPacketDeleter::operator()(AVPacket* p) {
  av_packet_unref(p);
  av_packet_free(&p);
}

void AVFrameDeleter::operator()(AVFrame* p) {
  av_frame_unref(p);
  av_frame_free(&p);
}

void AVBufferRefDeleter::operator()(AVBufferRef* p) {
  av_buffer_unref(&p);
}

void AVDictionaryDeleter::operator()(AVDictionary* p) {
  av_dict_free(&p);
}

} // namespace spdl
