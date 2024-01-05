#include <libspdl/detail/conversion.h>
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

int Frames::get_width() const {
  return frames.size() ? frames[0]->width : -1;
}

int Frames::get_height() const {
  return frames.size() ? frames[0]->height : -1;
}

int Frames::get_sample_rate() const {
  return frames.size() ? frames[0]->sample_rate : -1;
}

int Frames::get_num_samples() const {
  int ret = 0;
  for (auto& f : frames) {
    ret += f->nb_samples;
  }
  return ret;
}

Frames Frames::slice(int start, int stop, int step) const {
  Frames out{};
  for (int i = start; i < stop; i += step) {
    AVFrame* dst = CHECK_AVALLOCATE(av_frame_alloc());
    CHECK_AVERROR(
        av_frame_ref(dst, frames[i]),
        "Failed to create a new reference to an AVFrame.");
    out.frames.push_back(dst);
  }
  return out;
}

Buffer Frames::to_video_buffer(int plane) const {
  if (type != Type::Video) {
    SPDL_FAIL("Frames class is not video type.");
  }
  return detail::convert_video_frames(frames, plane);
}

} // namespace spdl
