#include <libspdl/core/frames.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#define _IS_AUDIO (media_type == MediaType::Audio)
#define _IS_VIDEO (media_type == MediaType::Video)
#define _IS_IMAGE (media_type == MediaType::Image)

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FFmpegFrames<media_type>::FFmpegFrames(uint64_t id_) : id(id_) {
  TRACE_EVENT(
      "decoding",
      "FFmpegFrames::FFmpegFrames",
      perfetto::Flow::ProcessScoped(id));
}

template <MediaType media_type>
FFmpegFrames<media_type>::FFmpegFrames(FFmpegFrames&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media_type>
FFmpegFrames<media_type>& FFmpegFrames<media_type>::operator=(
    FFmpegFrames<media_type>&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(frames, other.frames);
  return *this;
}

template <MediaType media_type>
FFmpegFrames<media_type>::~FFmpegFrames() {
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

template <MediaType media_type>
uint64_t FFmpegFrames<media_type>::get_id() const {
  return id;
}

template <MediaType media_type>
void FFmpegFrames<media_type>::push_back(AVFrame* frame) {
  if constexpr (media_type == MediaType::Image) {
    if (frames.size() > 0) {
      SPDL_FAIL_INTERNAL(
          "Attempted to store multiple frames to FFmpegImageFrames");
    }
  }
  frames.push_back(frame);
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_num_frames() const {
  if constexpr (_IS_AUDIO) {
    int ret = 0;
    for (auto& f : frames) {
      ret += f->nb_samples;
    }
    return ret;
  } else {
    return frames.size();
  }
}

template <MediaType media_type>
const std::vector<AVFrame*>& FFmpegFrames<media_type>::get_frames() const {
  return frames;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
int FFmpegFrames<media_type>::get_sample_rate() const requires _IS_AUDIO {
  return frames.size() ? frames[0]->sample_rate : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_num_channels() const requires _IS_AUDIO {
  return frames.size() ? frames[0]->channels : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
bool FFmpegFrames<media_type>::is_cuda() const
    requires(_IS_IMAGE || _IS_VIDEO) {
  if (!frames.size()) {
    return false;
  }
  return static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_num_planes() const
    requires(_IS_IMAGE || _IS_VIDEO) {
  return frames.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format)
      : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_width() const
    requires(_IS_IMAGE || _IS_VIDEO) {
  return frames.size() ? frames[0]->width : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_height() const
    requires(_IS_IMAGE || _IS_VIDEO) {
  return frames.size() ? frames[0]->height : -1;
}

//////////////////////////////////////////////////////////////////////////////
// Video specific
//////////////////////////////////////////////////////////////////////////////

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

template <MediaType media_type>
FFmpegVideoFrames FFmpegFrames<media_type>::slice(int start, int stop, int step)
    const requires _IS_VIDEO {
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

template <MediaType media_type>
FFmpegImageFrames FFmpegFrames<media_type>::slice(
    int i) const requires _IS_VIDEO {
  const int numel = frames.size();
  int stop = i + 1, step = 1;
  if (!adjust_indices(numel, &i, &stop, step)) {
    throw std::out_of_range(
        fmt::format("Index {} is outside of [0, {})", i, frames.size()));
  }
  auto out = FFmpegFrames<MediaType::Image>{id};
  assert(0 <= i && i < numel);
  out.push_back(detail::make_reference(frames[i]));
  return out;
}
template struct FFmpegFrames<MediaType::Audio>;
template struct FFmpegFrames<MediaType::Video>;
template struct FFmpegFrames<MediaType::Image>;

} // namespace spdl::core
