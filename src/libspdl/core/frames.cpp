/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frames.h>

#include <libspdl/core/rational_utils.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

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
Frames<media>::Frames(uintptr_t id, Rational time_base)
    : id_(id), time_base_(time_base) {
  TRACE_EVENT("decoding", "Frames::Frames", perfetto::Flow::ProcessScoped(id_));
  if (time_base_.den == 0) {
    SPDL_FAIL(
        fmt::format(
            "Invalid time base was provided. {}/{}",
            time_base_.num,
            time_base_.den));
  }
}

template <MediaType media>
Frames<media>::Frames(Frames&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media>
Frames<media>& Frames<media>::operator=(Frames<media>&& other) noexcept {
  using std::swap;
  swap(id_, other.id_);
  swap(frames_, other.frames_);
  return *this;
}

template <MediaType media>
Frames<media>::~Frames() {
  TRACE_EVENT(
      "decoding", "Frames::~Frames", perfetto::Flow::ProcessScoped(id_));
  std::for_each(frames_.begin(), frames_.end(), [](AVFrame* p) {
    DEBUG_PRINT_AVFRAME_REFCOUNT(p);
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

template <MediaType media>
uint64_t Frames<media>::get_id() const {
  return id_;
}

template <MediaType media>
void Frames<media>::push_back(AVFrame* frame) {
  if constexpr (media == MediaType::Image) {
    if (frames_.size() > 0) {
      SPDL_FAIL_INTERNAL("Attempted to store multiple frames to ImageFrames");
    }
  }
  frames_.push_back(frame);
}

template <MediaType media>
int64_t Frames<media>::get_pts(size_t index) const {
  if (auto num_frames = frames_.size(); index >= num_frames) {
    throw std::out_of_range(
        fmt::format("{} is out of range [0, {})", index, num_frames));
  }
  return frames_.at(index)->pts;
}

template <MediaType media>
double Frames<media>::get_timestamp(size_t index) const {
  return av_q2d(to_rational(get_pts(index), time_base_));
}

////////////////////////////////////////////////////////////////////////////////
// Common
////////////////////////////////////////////////////////////////////////////////

template <MediaType media>
const char* Frames<media>::get_media_format_name() const {
  if (frames_.size() == 0) {
    return "n/a";
  }
  return detail::get_media_format_name<media>(frames_[0]->format);
}

template <MediaType media>
OptionDict Frames<media>::get_metadata() const {
  if (frames_.size() == 0) {
    return {};
  }
  return detail::parse_dict(frames_[0]->metadata);
}

template <MediaType media>
int Frames<media>::get_num_frames() const {
  if constexpr (_IS_AUDIO) {
    int ret = 0;
    for (auto& f : frames_) {
      ret += f->nb_samples;
    }
    return ret;
  } else {
    return (int)frames_.size();
  }
}

template <MediaType media>
const std::vector<AVFrame*>& Frames<media>::get_frames() const {
  return frames_;
}

template <MediaType media>
FramesPtr<media> Frames<media>::clone() const {
  auto other = std::make_unique<Frames<media>>(id_, time_base_);
  for (const AVFrame* f : frames_) {
    other->push_back(CHECK_AVALLOCATE(av_frame_clone(f)));
  }
  return other;
}

template <MediaType media>
Rational Frames<media>::get_time_base() const {
  return time_base_;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
int Frames<media>::get_sample_rate() const
  requires _IS_AUDIO
{
  return frames_.size() ? frames_[0]->sample_rate : -1;
}

template <MediaType media>
int Frames<media>::get_num_channels() const
  requires _IS_AUDIO
{
  return frames_.size() ? GET_NUM_CHANNELS(frames_[0]) : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
int Frames<media>::get_num_planes() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames_.size()
      ? av_pix_fmt_count_planes((AVPixelFormat)frames_[0]->format)
      : -1;
}

template <MediaType media>
int Frames<media>::get_width() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames_.size() ? frames_[0]->width : -1;
}

template <MediaType media>
int Frames<media>::get_height() const
  requires(_IS_IMAGE || _IS_VIDEO)
{
  return frames_.size() ? frames_[0]->height : -1;
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
  const auto numel = (int)frames_.size();
  int len = adjust_indices(numel, &start, &stop, step);

  auto out = std::make_unique<VideoFrames>(id_, time_base_);
  if (!len) {
    return out;
  }

  for (int i = start; i < stop; i += step) {
    assert(0 <= i && i < numel);
    out->frames_.push_back(detail::make_reference(frames_.at(i)));
  }
  return out;
}

template <MediaType media>
VideoFramesPtr Frames<media>::slice(const std::vector<int64_t>& index) const
  requires _IS_VIDEO
{
  const auto numel = (int)frames_.size();
  for (const auto& i : index) {
    if (i >= numel || i < -numel) {
      throw std::out_of_range(
          fmt::format("Index {} is outside of [0, {})", i, frames_.size()));
    }
  }
  auto out = std::make_unique<VideoFrames>(id_, time_base_);
  if (!index.size()) {
    return out;
  }
  for (auto i : index) {
    if (i < 0) {
      i += numel;
    }
    out->frames_.push_back(detail::make_reference(frames_.at(i)));
  }
  return out;
}

template <MediaType media>
ImageFramesPtr Frames<media>::slice(int64_t i) const
  requires _IS_VIDEO
{
  const auto numel = (int)frames_.size();
  if (i >= numel || i < -numel) {
    throw std::out_of_range(
        fmt::format("Index {} is outside of [0, {})", i, numel));
  }
  if (i < 0) {
    i += numel;
  }
  assert(0 <= i && i < numel);
  auto out = std::make_unique<Frames<MediaType::Image>>(id_, time_base_);
  out->push_back(detail::make_reference(frames_.at(i)));
  return out;
}

template class Frames<MediaType::Audio>;
template class Frames<MediaType::Video>;
template class Frames<MediaType::Image>;

} // namespace spdl::core
