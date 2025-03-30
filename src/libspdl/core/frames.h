/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

struct AVFrame;

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Frames
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
class FFmpegFrames;

using FFmpegAudioFrames = FFmpegFrames<MediaType::Audio>;
using FFmpegVideoFrames = FFmpegFrames<MediaType::Video>;
using FFmpegImageFrames = FFmpegFrames<MediaType::Image>;

template <MediaType media_type>
using FFmpegFramesPtr = std::unique_ptr<FFmpegFrames<media_type>>;

using FFmpegAudioFramesPtr = FFmpegFramesPtr<MediaType::Audio>;
using FFmpegVideoFramesPtr = FFmpegFramesPtr<MediaType::Video>;
using FFmpegImageFramesPtr = FFmpegFramesPtr<MediaType::Image>;

#define _IS_AUDIO (media_type == MediaType::Audio)
#define _IS_VIDEO (media_type == MediaType::Video)
#define _IS_IMAGE (media_type == MediaType::Image)

///
/// Base class that holds media frames decoded with FFmpeg.
template <MediaType media_type>
class FFmpegFrames {
 private:
  ///
  /// Used for tracking the lifetime in tracing.
  uint64_t id{0};

  /// Time base of the frames
  Rational time_base;

 private:
  /// Series of decoded frames generated by FFmpeg.
  /// If media type is image, there will be exactly one ``AVFrame`` instance.
  /// If media type is video, there will be multiple of ``AVFrame`` instances
  /// and each of which represents one frame of video. If media type is audio,
  /// there will be multiple of ``AVFrame`` instances and each contains multiple
  /// audio samples.
  ///
  /// We deal with multiple frames at a time, so we use vector of raw
  /// pointers with dedicated destructor, as opposed to vector of managed
  /// pointers
  std::vector<AVFrame*> frames{};

 public:
  FFmpegFrames(uint64_t id, Rational time_base);

  ///
  /// No copy constructor
  FFmpegFrames(const FFmpegFrames&) = delete;
  ///
  /// No copy assignment operator
  FFmpegFrames& operator=(const FFmpegFrames&) = delete;
  ///
  /// Move constructor
  FFmpegFrames(FFmpegFrames&&) noexcept;
  ///
  /// Move assignment operator
  FFmpegFrames& operator=(FFmpegFrames&&) noexcept;
  ///
  /// Destructor releases ``AVFrame`` resources
  ~FFmpegFrames();

  ///
  /// Get the ID used for tracing.
  uint64_t get_id() const;

  ///
  /// Get the list of frames.
  const std::vector<AVFrame*>& get_frames() const;

  //////////////////////////////////////////////////////////////////////////////
  // Common
  //////////////////////////////////////////////////////////////////////////////

  ///
  /// Get the format of the frames.
  const char* get_media_format_name() const;

  ///
  /// Get metadata
  OptionDict get_metadata() const;

  ///
  /// Get the number of frames.
  int get_num_frames() const;
  // the behavior is different for audio

  ///
  /// Get the time_base, which is the unit of time that this Frame uses.
  Rational get_time_base() const;

  ///
  /// Push a new frame into the container.
  void push_back(AVFrame* frame);
  // the behavior is different for image

  // Get the PTS of the specified frame.
  // throws if the index is not within the range
  int64_t get_pts(size_t index = 0) const;

  FFmpegFramesPtr<media_type> clone() const;

  //////////////////////////////////////////////////////////////////////////////
  // Audio specific
  //////////////////////////////////////////////////////////////////////////////

  ///
  /// Get the sample rate
  int get_sample_rate() const
    requires _IS_AUDIO;

  ///
  /// Get the number of audio channels.
  int get_num_channels() const
    requires _IS_AUDIO;

  //////////////////////////////////////////////////////////////////////////////
  // Common to Image/Video
  //////////////////////////////////////////////////////////////////////////////

  /// Get the number of planes in the image.
  ///
  /// Note: The number of planes and the number of color channels do not match.
  /// For example, NV12 has 3 channels, YUV, but U and V are interleaved in the
  /// same plane.
  int get_num_planes() const
    requires(_IS_IMAGE || _IS_VIDEO);

  ///
  /// Get the width of the image.
  int get_width() const
    requires(_IS_IMAGE || _IS_VIDEO);

  ///
  /// Get the height of the image.
  int get_height() const
    requires(_IS_IMAGE || _IS_VIDEO);

  //////////////////////////////////////////////////////////////////////////////
  // Video specific
  //////////////////////////////////////////////////////////////////////////////

  ///
  /// Range slice operation, using Python's slice notation.
  FFmpegVideoFramesPtr slice(int start, int stop, int step) const
    requires _IS_VIDEO;

  ///
  /// Slice (`__getitem__`) operation.
  FFmpegVideoFramesPtr slice(const std::vector<int64_t>& index) const
    requires _IS_VIDEO;

  ///
  /// Slice (`__getitem__`) operation.
  FFmpegImageFramesPtr slice(int64_t index) const
    requires _IS_VIDEO;
};

template <MediaType media_type>
FFmpegFramesPtr<media_type> clone(const FFmpegFrames<media_type>& src);

#undef _IS_AUDIO
#undef _IS_VIDEO
#undef _IS_IMAGE

} // namespace spdl::core
