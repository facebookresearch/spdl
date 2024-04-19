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

template <MediaType media_type>
struct NvDecFrames;

using NvDecVideoFrames = NvDecFrames<MediaType::Video>;
using NvDecImageFrames = NvDecFrames<MediaType::Image>;

template <MediaType media_type>
using NvDecFramesPtr = std::unique_ptr<NvDecFrames<media_type>>;

using NvDecVideoFramesPtr = NvDecFramesPtr<MediaType::Video>;
using NvDecImageFramesPtr = NvDecFramesPtr<MediaType::Image>;

// Wrapper for Python
template <MediaType media_type, template <MediaType> typename FramesPtr>
class FramesWrapper;

template <MediaType media_type, template <MediaType> typename FramesPtr>
using FramesWrapperPtr = std::shared_ptr<FramesWrapper<media_type, FramesPtr>>;

template <MediaType media_type>
using FFmpegFramesWrapperPtr = FramesWrapperPtr<media_type, FFmpegFramesPtr>;

using FFmpegAudioFramesWrapperPtr = FFmpegFramesWrapperPtr<MediaType::Audio>;
using FFmpegVideoFramesWrapperPtr = FFmpegFramesWrapperPtr<MediaType::Video>;
using FFmpegImageFramesWrapperPtr = FFmpegFramesWrapperPtr<MediaType::Image>;

template <MediaType media_type>
using NvDecFramesWrapperPtr = FramesWrapperPtr<media_type, NvDecFramesPtr>;

using NvDecVideoFramesWrapperPtr = NvDecFramesWrapperPtr<MediaType::Video>;
using NvDecImageFramesWrapperPtr = NvDecFramesWrapperPtr<MediaType::Image>;

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

 public:
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
  /// Get the number of frames.
  int get_num_frames() const;
  // the behavior is different for audio

  ///
  /// Push a new frame into the container.
  void push_back(AVFrame* frame);
  // the behavior is different for image

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
  // Common to Audio/Video
  //////////////////////////////////////////////////////////////////////////////
  Rational get_time_base() const
    requires(_IS_AUDIO || _IS_VIDEO);

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
  FFmpegImageFramesPtr slice(int index) const
    requires _IS_VIDEO;
};

#undef _IS_AUDIO
#undef _IS_VIDEO
#undef _IS_IMAGE

////////////////////////////////////////////////////////////////////////////////
// NVDEC - Video
////////////////////////////////////////////////////////////////////////////////

/// Class that holds media frames decoded with NVDEC decoder.
/// The decoded media can be video or image.
template <MediaType media_type>
struct NvDecFrames {
#ifdef SPDL_USE_NVCODEC
 private:
  ///
  /// Used for tracking the lifetime in tracing.
  uint64_t id{0};

 public:
  /// ``enum AVPixelFormat`` but using ``int`` so as to avoid including FFmpeg
  /// header.
  int media_format;

  /// The data buffer. Because when using NVDEC decoder, we need to directly
  /// copy the decoded frame from decoder's output buffer, we use a continuous
  /// memory buffer to store the data.
  CUDABuffer2DPitchPtr buffer;

  NvDecFrames(uint64_t id, int media_format);
  NvDecFrames(const NvDecFrames&) = delete;
  NvDecFrames& operator=(const NvDecFrames&) = delete;
  NvDecFrames(NvDecFrames&&) noexcept;
  NvDecFrames& operator=(NvDecFrames&&) noexcept;
  ~NvDecFrames() = default;

  const char* get_media_format_name() const;

  uint64_t get_id() const {
    return id;
  }
#endif
};

////////////////////////////////////////////////////////////////////////////////
// FramesWrapper
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, template <MediaType> typename FramesPtr>
class FramesWrapper {
 protected:
  FramesPtr<media_type> frames;

 public:
  FramesWrapper(FramesPtr<media_type>&& p) : frames(std::move(p)){};

  FramesPtr<media_type> unwrap() {
    if (!frames) {
      throw std::runtime_error(
          "Frames is in invalid state. Perhaps it's already released?");
    }
    return std::move(frames);
  }

  int get_id() const {
    if (!frames) {
      throw std::runtime_error(
          "Frames is in invalid state. Perhaps it's already released?");
    }
    return frames->get_id();
  }

  const FramesPtr<media_type>& get_frames_ref() const {
    if (!frames) {
      throw std::runtime_error(
          "Frames is in invalid state. Perhaps it's already released?");
    }
    return frames;
  }
};

template <MediaType media_type, template <MediaType> typename FramesPtr>
FramesWrapperPtr<media_type, FramesPtr> wrap(FramesPtr<media_type>&& frames) {
  return std::make_shared<FramesWrapper<media_type, FramesPtr>>(
      std::move(frames));
}

template <MediaType media_type, template <MediaType> typename FramesPtr>
std::vector<FramesWrapperPtr<media_type, FramesPtr>> wrap(
    std::vector<FramesPtr<media_type>>&& frames) {
  std::vector<FramesWrapperPtr<media_type, FramesPtr>> ret;

  for (auto& frame : frames) {
    ret.push_back(wrap<media_type, FramesPtr>(std::move(frame)));
  }
  return ret;
}
} // namespace spdl::core
