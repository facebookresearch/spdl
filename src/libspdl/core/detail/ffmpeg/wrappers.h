#pragma once

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

namespace spdl::core::detail {
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

// AVBSFContext
struct AVBSFContextDeleter {
  void operator()(AVBSFContext* p);
};

using AVBSFContextPtr = std::unique_ptr<AVBSFContext, AVBSFContextDeleter>;

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

} // namespace spdl::core::detail

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

#ifdef SPDL_DEBUG_REFCOUNT
namespace spdl::core::detail {
void debug_log_avframe_refcount(AVFrame* frame);
void debug_log_avpacket_refcount(AVPacket* packet);
} // namespace spdl::core::detail
#define DEBUG_PRINT_AVFRAME_REFCOUNT(x) \
  ::spdl::core::detail::debug_log_avframe_refcount(x)
#define DEBUG_PRINT_AVPACKET_REFCOUNT(x) \
  ::spdl::core::detail::debug_log_avpacket_refcount(x)
#else
#define DEBUG_PRINT_AVFRAME_REFCOUNT(x)
#define DEBUG_PRINT_AVPACKET_REFCOUNT(x)
#endif
