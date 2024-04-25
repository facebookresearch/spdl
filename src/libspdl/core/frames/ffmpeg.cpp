#include <libspdl/core/frames.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <cassert>
#include <exception>

#include <folly/logging/xlog.h>

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
FFmpegFrames<media_type>::FFmpegFrames(uint64_t id_, Rational time_base_)
    : id(id_), time_base(time_base_) {
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

////////////////////////////////////////////////////////////////////////////////
// Common
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
const char* FFmpegFrames<media_type>::get_media_format_name() const {
  if (frames.size() == 0) {
    return "n/a";
  }
  return detail::get_media_format_name<media_type>(frames[0]->format);
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
// Common to Audio/Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
Rational FFmpegFrames<media_type>::get_time_base() const
  requires(_IS_AUDIO || _IS_VIDEO)
{
  return time_base;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
int FFmpegFrames<media_type>::get_sample_rate() const
  requires _IS_AUDIO
{
  return frames.size() ? frames[0]->sample_rate : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_num_channels() const
  requires _IS_AUDIO
{
  return frames.size() ?
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 2, 100)
                       frames[0]->ch_layout.nb_channels
#else
                       frames[0]->channels
#endif
                       : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
int FFmpegFrames<media_type>::get_num_planes() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format)
      : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_width() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames.size() ? frames[0]->width : -1;
}

template <MediaType media_type>
int FFmpegFrames<media_type>::get_height() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
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
FFmpegVideoFramesPtr
FFmpegFrames<media_type>::slice(int start, int stop, int step) const
  requires _IS_VIDEO
{
  const int numel = frames.size();
  int len = adjust_indices(numel, &start, &stop, step);

  auto out = std::make_unique<FFmpegVideoFrames>(id, time_base);
  if (!len) {
    return out;
  }

  for (int i = start; i < stop; i += step) {
    assert(0 <= i && i < numel);
    out->frames.push_back(detail::make_reference(frames[i]));
  }
  return out;
}

template <MediaType media_type>
FFmpegVideoFramesPtr FFmpegFrames<media_type>::slice(
    const std::vector<int64_t> index) const
  requires _IS_VIDEO
{
  const int numel = frames.size();
  for (const auto& i : index) {
    if (i >= numel || i < -numel) {
      throw std::out_of_range(
          fmt::format("Index {} is outside of [0, {})", i, frames.size()));
    }
  }
  auto out = std::make_unique<FFmpegVideoFrames>(id, time_base);
  if (!index.size()) {
    return out;
  }
  for (auto i : index) {
    if (i < 0) {
      i += numel;
    }
    out->frames.push_back(detail::make_reference(frames[i]));
  }
  return out;
}

template <MediaType media_type>
FFmpegImageFramesPtr FFmpegFrames<media_type>::slice(int64_t i) const
  requires _IS_VIDEO
{
  const int numel = static_cast<int64_t>(frames.size());
  if (i >= numel || i < -numel) {
    throw std::out_of_range(
        fmt::format("Index {} is outside of [0, {})", i, numel));
  }
  if (i < 0) {
    i += numel;
  }
  assert(0 <= i && i < numel);
  auto out = std::make_unique<FFmpegFrames<MediaType::Image>>(id, time_base);
  out->push_back(detail::make_reference(frames[i]));
  return out;
}

template struct FFmpegFrames<MediaType::Audio>;
template struct FFmpegFrames<MediaType::Video>;
template struct FFmpegFrames<MediaType::Image>;

template <MediaType media_type>
FFmpegFramesPtr<media_type> clone(const FFmpegFrames<media_type>& src) {
  auto other =
      std::make_unique<FFmpegFrames<media_type>>(src.get_id(), src.time_base);
  for (const AVFrame* f : src.get_frames()) {
    other->push_back(CHECK_AVALLOCATE(av_frame_clone(f)));
  }
  return other;
}

template FFmpegAudioFramesPtr clone(const FFmpegAudioFrames&);
template FFmpegVideoFramesPtr clone(const FFmpegVideoFrames&);
template FFmpegImageFramesPtr clone(const FFmpegImageFrames&);

} // namespace spdl::core
