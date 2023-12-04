#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

#include <memory>

namespace spdl {
////////////////////////////////////////////////////////////////////////////////
// RAII wrappers
////////////////////////////////////////////////////////////////////////////////

// AVIOContext
struct AVIOContextDeleter {
  void operator()(AVIOContext* p);
};

using AVIOContextPtr = std::unique_ptr<AVIOContext, AVIOContextDeleter>;

// AVFormatContext (read)
struct AVFormatInputContextDeleter {
  void operator()(AVFormatContext* p);
};

using AVFormatInputContextPtr =
    std::unique_ptr<AVFormatContext, AVFormatInputContextDeleter>;

// AVCodecContext
struct AVCodecContextDeleter {
  void operator()(AVCodecContext* p);
};

using AVCodecContextPtr =
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

// AVPacket
// Assumption: AVPacket is reference counted and the counter is always 1.
struct AVPacketDeleter {
  void operator()(AVPacket* p);
};

using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;

// AVFrame
// Assumption: AVFrame is reference counted and the counter is always 1.
struct AVFrameDeleter {
  void operator()(AVFrame* p);
};

using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

// AutoBufferRef
struct AVBufferRefDeleter {
  void operator()(AVBufferRef* p);
};

using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;

// AVDictionary
struct AVDictionaryDeleter {
  void operator()(AVDictionary* p);
};

using AVDictionaryPtr = std::unique_ptr<AVDictionary, AVDictionaryDeleter>;

} // namespace spdl
