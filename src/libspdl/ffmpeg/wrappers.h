#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
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

// AVFilterGraph
struct AVFilterGraphDeleter {
  void operator()(AVFilterGraph* p);
};

using AVFilterGraphPtr = std::unique_ptr<AVFilterGraph, AVFilterGraphDeleter>;

// AVFrameAutoUnref
struct AVFrameAutoUnref {
  AVFrame* p;
  AVFrameAutoUnref(AVFrame* p);
  ~AVFrameAutoUnref();
};

// RAII wrapper for objects that require clean up, but need to expose double
// pointer. As they are re-allocated by FFmpeg functions.
// Examples are AVDictionary and AVFilterInOut.
// They are typically only used in cpp files, so we use macro and let each
// implementation file defines them in anonymous scope.
#define DEF_DPtr(Type, delete_func)                         \
  class Type##DPtr {                                        \
    Type* p = nullptr;                                      \
                                                            \
   public:                                                  \
    explicit Type##DPtr(Type* p_ = nullptr) : p(p_) {}      \
    Type##DPtr(const Type##DPtr&) = delete;                 \
    Type##DPtr& operator=(const Type##DPtr&) = delete;      \
    Type##DPtr(Type##DPtr&&) noexcept = default;            \
    Type##DPtr& operator=(Type##DPtr&&) noexcept = default; \
    ~Type##DPtr() {                                         \
      delete_func(&p);                                      \
    };                                                      \
    operator Type*() const {                                \
      return p;                                             \
    };                                                      \
    operator Type**() {                                     \
      return &p;                                            \
    };                                                      \
    Type* operator->() const {                              \
      return p;                                             \
    };                                                      \
  }

} // namespace spdl
