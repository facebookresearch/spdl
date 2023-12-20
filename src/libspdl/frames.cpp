#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/frames.h>

extern "C" {
#include <libavutil/frame.h>
}

#ifdef SPDL_DEBUG_REFCOUNT
#include <stdatomic.h>
static struct AVBuffer {
  uint8_t* data;
  size_t size;
  atomic_uint refcount;
  void (*free)(void* opaque, uint8_t* data);
  void* opaque;
  int flags;
  int flags_internal;
};
#endif

namespace spdl {

Frames::~Frames() {
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
#ifdef SPDL_DEBUG_REFCOUNT
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
#endif
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

Frames slice_frames(const Frames& orig, int start, int stop, int step) {
  Frames out{};
  for (int i = start; i < stop; i += step) {
    AVFrame* dst = CHECK_AVALLOCATE(av_frame_alloc());
    CHECK_AVERROR(
        av_frame_ref(dst, orig.frames[i]),
        "Failed to create a new reference to an AVFrame.");
    out.frames.push_back(dst);
  }
  return out;
}

} // namespace spdl
