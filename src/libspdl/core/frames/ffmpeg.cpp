#include <libspdl/core/frames.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
FFmpegFrames::FFmpegFrames(uint64_t id_) : id(id_) {
  TRACE_EVENT(
      "decoding",
      "FFmpegFrames::FFmpegFrames",
      perfetto::Flow::ProcessScoped(id));
}

FFmpegFrames::FFmpegFrames(FFmpegFrames&& other) noexcept {
  *this = std::move(other);
}

FFmpegFrames& FFmpegFrames::operator=(FFmpegFrames&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(frames, other.frames);
  return *this;
}

FFmpegFrames::~FFmpegFrames() {
  TRACE_EVENT(
      "decoding",
      "FFmpegFrames::~FFmpegFrames",
      perfetto::Flow::ProcessScoped(id));
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    DEBUG_PRINT_AVFRAME_REFCOUNT(p);
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

uint64_t FFmpegFrames::get_id() const {
  return id;
}

void FFmpegFrames::push_back(AVFrame* frame) {
  frames.push_back(frame);
}

int FFmpegFrames::get_num_frames() const {
  return frames.size();
}

const std::vector<AVFrame*>& FFmpegFrames::get_frames() const {
  return frames;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
enum MediaType FFmpegAudioFrames::get_media_type() const {
  return MediaType::Audio;
}

int FFmpegAudioFrames::get_sample_rate() const {
  return frames.size() ? frames[0]->sample_rate : -1;
}

int FFmpegAudioFrames::get_num_frames() const {
  int ret = 0;
  for (auto& f : frames) {
    ret += f->nb_samples;
  }
  return ret;
}

int FFmpegAudioFrames::get_num_channels() const {
  return frames.size() ? frames[0]->channels : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
enum MediaType FFmpegVideoFrames::get_media_type() const {
  return MediaType::Video;
}

bool FFmpegVideoFrames::is_cuda() const {
  if (!frames.size()) {
    return false;
  }
  return static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
}

int FFmpegVideoFrames::get_num_planes() const {
  return frames.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format)
      : -1;
}

int FFmpegVideoFrames::get_width() const {
  return frames.size() ? frames[0]->width : -1;
}

int FFmpegVideoFrames::get_height() const {
  return frames.size() ? frames[0]->height : -1;
}

namespace {
int adjust_indices(const int length, int* start, int* stop, int step) {
  if (step <= 0) {
    SPDL_FAIL(fmt::format("Step must be larget than 0. Found: {}", step));
  }
  if (*start < 0) {
    *start += length;
    if (*start < 0) {
      *start = (step < 0) ? -1 : 0;
    }
  } else if (*start >= length) {
    *start = (step < 0) ? length - 1 : length;
  }

  if (*stop < 0) {
    *stop += length;
    if (*stop < 0) {
      *stop = (step < 0) ? -1 : 0;
    }
  } else if (*stop >= length) {
    *stop = (step < 0) ? length - 1 : length;
  }

  if (step < 0) {
    if (*stop < *start) {
      return (*start - *stop - 1) / (-step) + 1;
    }
  } else {
    if (*start < *stop) {
      return (*stop - *start - 1) / step + 1;
    }
  }
  return 0;
}
} // namespace

FFmpegVideoFrames FFmpegVideoFrames::slice(int start, int stop, int step)
    const {
  const int numel = frames.size();
  int len = adjust_indices(numel, &start, &stop, step);

  auto out = FFmpegVideoFrames{id};
  if (!len) {
    return out;
  }

  for (int i = start; i < stop; i += step) {
    assert(0 <= i && i < numel);
    out.frames.push_back(detail::make_reference(frames[i]));
  }
  return out;
}

FFmpegImageFrames FFmpegVideoFrames::slice(int i) const {
  const int numel = frames.size();
  int stop = i + 1, step = 1;
  if (!adjust_indices(numel, &i, &stop, step)) {
    throw std::out_of_range(
        fmt::format("Index {} is outside of [0, {})", i, frames.size()));
  }
  auto out = FFmpegImageFrames{id};
  assert(0 <= i && i < numel);
  out.push_back(detail::make_reference(frames[i]));
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Image
////////////////////////////////////////////////////////////////////////////////
enum MediaType FFmpegImageFrames::get_media_type() const {
  return MediaType::Image;
}

bool FFmpegImageFrames::is_cuda() const {
  return false;
}

int FFmpegImageFrames::get_num_planes() const {
  return frames.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format)
      : -1;
}

int FFmpegImageFrames::get_width() const {
  return frames.size() ? frames[0]->width : -1;
}

int FFmpegImageFrames::get_height() const {
  return frames.size() ? frames[0]->height : -1;
}

void FFmpegImageFrames::push_back(AVFrame* frame) {
  if (frames.size() > 0) {
    SPDL_FAIL_INTERNAL(
        "Attempted to store multiple frames to FFmpegImageFrames");
  }
  FFmpegFrames::push_back(frame);
}

} // namespace spdl::core
