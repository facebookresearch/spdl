/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frames.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <algorithm>
#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#define _IS_AUDIO (media == MediaType::Audio)
#define _IS_VIDEO (media == MediaType::Video)
#define _IS_IMAGE (media == MediaType::Image)

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
Frames<media>::Frames(uintptr_t id_, Rational time_base_)
    : id(id_), time_base(time_base_) {
  TRACE_EVENT("decoding", "Frames::Frames", perfetto::Flow::ProcessScoped(id));
  if (time_base.den == 0) {
    SPDL_FAIL(fmt::format(
        "Invalid time base was provided. {}/{}", time_base.num, time_base.den));
  }
}

template <MediaType media>
Frames<media>::Frames(Frames&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media>
Frames<media>& Frames<media>::operator=(Frames<media>&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(frames, other.frames);
  return *this;
}

template <MediaType media>
Frames<media>::~Frames() {
  TRACE_EVENT("decoding", "Frames::~Frames", perfetto::Flow::ProcessScoped(id));
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    DEBUG_PRINT_AVFRAME_REFCOUNT(p);
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

template <MediaType media>
uint64_t Frames<media>::get_id() const {
  return id;
}

template <MediaType media>
void Frames<media>::push_back(AVFrame* frame) {
  if constexpr (media == MediaType::Image) {
    if (frames.size() > 0) {
      SPDL_FAIL_INTERNAL("Attempted to store multiple frames to ImageFrames");
    }
  }
  frames.push_back(frame);
}

template <MediaType media>
int64_t Frames<media>::get_pts(size_t index) const {
  auto num_frames = frames.size();
  if (index >= num_frames) {
    throw std::out_of_range(
        fmt::format("{} is out of range [0, {})", index, num_frames));
  }
  return frames.at(index)->pts;
}

////////////////////////////////////////////////////////////////////////////////
// Common
////////////////////////////////////////////////////////////////////////////////

template <MediaType media>
const char* Frames<media>::get_media_format_name() const {
  if (frames.size() == 0) {
    return "n/a";
  }
  return detail::get_media_format_name<media>(frames[0]->format);
}

template <MediaType media>
OptionDict Frames<media>::get_metadata() const {
  if (frames.size() == 0) {
    return {};
  }
  return detail::parse_dict(frames[0]->metadata);
}

template <MediaType media>
int Frames<media>::get_num_frames() const {
  if constexpr (_IS_AUDIO) {
    int ret = 0;
    for (auto& f : frames) {
      ret += f->nb_samples;
    }
    return ret;
  } else {
    return (int)frames.size();
  }
}

template <MediaType media>
const std::vector<AVFrame*>& Frames<media>::get_frames() const {
  return frames;
}

template <MediaType media>
FramesPtr<media> Frames<media>::clone() const {
  auto other = std::make_unique<Frames<media>>(id, time_base);
  for (const AVFrame* f : frames) {
    other->push_back(CHECK_AVALLOCATE(av_frame_clone(f)));
  }
  return other;
}

template <MediaType media>
Rational Frames<media>::get_time_base() const {
  return time_base;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
int Frames<media>::get_sample_rate() const
  requires _IS_AUDIO
{
  return frames.size() ? frames[0]->sample_rate : -1;
}

template <MediaType media>
int Frames<media>::get_num_channels() const
  requires _IS_AUDIO
{
  return frames.size() ? GET_NUM_CHANNELS(frames[0]) : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
int Frames<media>::get_num_planes() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format)
      : -1;
}

template <MediaType media>
int Frames<media>::get_width() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames.size() ? frames[0]->width : -1;
}

template <MediaType media>
int Frames<media>::get_height() const
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

template <MediaType media>
VideoFramesPtr Frames<media>::slice(int start, int stop, int step) const
  requires _IS_VIDEO
{
  const auto numel = (int)frames.size();
  int len = adjust_indices(numel, &start, &stop, step);

  auto out = std::make_unique<VideoFrames>(id, time_base);
  if (!len) {
    return out;
  }

  for (int i = start; i < stop; i += step) {
    assert(0 <= i && i < numel);
    out->frames.push_back(detail::make_reference(frames.at(i)));
  }
  return out;
}

template <MediaType media>
VideoFramesPtr Frames<media>::slice(const std::vector<int64_t>& index) const
  requires _IS_VIDEO
{
  const auto numel = (int)frames.size();
  for (const auto& i : index) {
    if (i >= numel || i < -numel) {
      throw std::out_of_range(
          fmt::format("Index {} is outside of [0, {})", i, frames.size()));
    }
  }
  auto out = std::make_unique<VideoFrames>(id, time_base);
  if (!index.size()) {
    return out;
  }
  for (auto i : index) {
    if (i < 0) {
      i += numel;
    }
    out->frames.push_back(detail::make_reference(frames.at(i)));
  }
  return out;
}

template <MediaType media>
ImageFramesPtr Frames<media>::slice(int64_t i) const
  requires _IS_VIDEO
{
  const auto numel = (int)frames.size();
  if (i >= numel || i < -numel) {
    throw std::out_of_range(
        fmt::format("Index {} is outside of [0, {})", i, numel));
  }
  if (i < 0) {
    i += numel;
  }
  assert(0 <= i && i < numel);
  auto out = std::make_unique<Frames<MediaType::Image>>(id, time_base);
  out->push_back(detail::make_reference(frames.at(i)));
  return out;
}

template class Frames<MediaType::Audio>;
template class Frames<MediaType::Video>;
template class Frames<MediaType::Image>;

} // namespace spdl::core
