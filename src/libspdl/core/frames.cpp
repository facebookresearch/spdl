#include <libspdl/core/frames.h>

#include <libspdl/core/detail/ffmpeg/logging.h>
#include <libspdl/core/detail/ffmpeg/wrappers.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {

FrameContainer::FrameContainer(MediaType type_) : type(type_) {}

FrameContainer::~FrameContainer() {
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    DEBUG_PRINT_AVFRAME_REFCOUNT(p);
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

bool FrameContainer::is_cuda() const {
  if (type == MediaType::Audio || !frames.size()) {
    return false;
  }
  return static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
}

std::string FrameContainer::get_format() const {
  if (!frames.size()) {
    return "none";
  }
  switch (type) {
    case MediaType::Audio:
      return av_get_sample_fmt_name((AVSampleFormat)frames[0]->format);
    case MediaType::Video:
      return av_get_pix_fmt_name((AVPixelFormat)frames[0]->format);
  }
}

int FrameContainer::get_num_planes() const {
  if (type != MediaType::Video) {
    SPDL_FAIL("FrameContainer are not video.");
  }
  return av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format);
}

int FrameContainer::get_width() const {
  return frames.size() ? frames[0]->width : -1;
}

int FrameContainer::get_height() const {
  return frames.size() ? frames[0]->height : -1;
}

int FrameContainer::get_sample_rate() const {
  return frames.size() ? frames[0]->sample_rate : -1;
}

int FrameContainer::get_num_samples() const {
  int ret = 0;
  for (auto& f : frames) {
    ret += f->nb_samples;
  }
  return ret;
}

FrameContainer FrameContainer::slice(int start, int stop, int step) const {
  FrameContainer out{type};
  for (int i = start; i < stop; i += step) {
    AVFrame* dst = CHECK_AVALLOCATE(av_frame_alloc());
    CHECK_AVERROR(
        av_frame_ref(dst, frames[i]),
        "Failed to create a new reference to an AVFrame.");
    out.frames.push_back(dst);
  }
  return out;
}

} // namespace spdl::core
