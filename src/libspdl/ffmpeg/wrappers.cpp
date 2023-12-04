#include <libspdl/ffmpeg/logging.h>
#include <libspdl/ffmpeg/wrappers.h>

namespace spdl {

////////////////////////////////////////////////////////////////////////////////
// RAII wrappers
////////////////////////////////////////////////////////////////////////////////

// AVIOContext
void AVIOContextDeleter::operator()(AVIOContext* p) {
  if (p) {
    avio_flush(p);
    av_freep(&p->buffer);
  }
  av_freep(&p);
};

// AVFormatContext
void AVFormatInputContextDeleter::operator()(AVFormatContext* p) {
  avformat_close_input(&p);
};

// AVCodecContext
void AVCodecContextDeleter::operator()(AVCodecContext* p) {
  avcodec_free_context(&p);
}

// AVPacket
void AVPacketDeleter::operator()(AVPacket* p) {
  av_packet_unref(p);
  av_packet_free(&p);
}

// AVFrame
void AVFrameDeleter::operator()(AVFrame* p) {
  av_frame_unref(p);
  av_frame_free(&p);
}

// AutoBufferRef
void AVBufferRefDeleter::operator()(AVBufferRef* p) {
  av_buffer_unref(&p);
}

// AVDictionary
void AVDictionaryDeleter::operator()(AVDictionary* p) {
  av_dict_free(&p);
}

} // namespace spdl
