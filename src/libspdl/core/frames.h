#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

struct AVFrame;

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Base
////////////////////////////////////////////////////////////////////////////////

///
/// Abstract class that represents the decoded media frames.
struct DecodedFrames {
  virtual ~DecodedFrames() = default;

  ///
  /// Type of stored media, such as Audio, Video and Image.
  virtual enum MediaType get_media_type() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////

///
/// Base class that holds media frames decoded with FFmpeg.
struct FFmpegFrames : public DecodedFrames {
  ///
  /// Used for tracking the lifetime in tracing.
  uint64_t id{0};

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

  FFmpegFrames(uint64_t id);
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
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////

/// Holds audio frames decoded with FFmpeg.
struct FFmpegAudioFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  ///
  /// Audio
  enum MediaType get_media_type() const override;

  ///
  /// Get the sample rate
  int get_sample_rate() const;

  ///
  /// Get the number of total frames.
  int get_num_frames() const;

  ///
  /// Get the number of audio channels.
  int get_num_channels() const;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Image
////////////////////////////////////////////////////////////////////////////////

/// Holds single image frame decoded with FFmpeg.
struct FFmpegImageFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  ///
  /// Image
  enum MediaType get_media_type() const override;

  ///
  /// True if the underlying image frame is on CUDA device.
  bool is_cuda() const;

  /// Get the number of planes in the image.
  ///
  /// Note: The number of planes and the number of color channels do not match.
  /// For example, NV12 has 3 channels, YUV, but U and V are interleaved in the
  /// same plane.
  int get_num_planes() const;

  ///
  /// Get the width of the image.
  int get_width() const;

  ///
  /// Get the height of the image.
  int get_height() const;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////

/// Holds video frame decoded with FFmpeg.
struct FFmpegVideoFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  ///
  /// Video
  enum MediaType get_media_type() const override;

  ///
  /// True if the underlying image frame is on CUDA device.
  bool is_cuda() const;

  ///
  /// Get the number of frames.
  int get_num_frames() const;

  /// Get the number of planes in the image.
  ///
  /// Note: The number of planes and the number of color channels do not match.
  /// For example, NV12 has 3 channels, YUV, but U and V are interleaved in the
  /// same plane.
  int get_num_planes() const;

  ///
  /// Get the width of the image
  int get_width() const;

  ///
  /// Get the height of the image
  int get_height() const;

  ///
  /// Range slice operation, using Python's slice notation.
  FFmpegVideoFrames slice(int start, int stop, int step) const;

  ///
  /// Slice (`__getitem__`) operation.
  FFmpegImageFrames slice(int index) const;
};

////////////////////////////////////////////////////////////////////////////////
// NVDEC - Video
////////////////////////////////////////////////////////////////////////////////

/// Class that holds media frames decoded with NVDEC decoder.
/// The decoded media can be video or image.
struct NvDecVideoFrames : public DecodedFrames {
#ifdef SPDL_USE_NVDEC
  ///
  /// Used for tracking the lifetime in tracing.
  uint64_t id{0};

  ///
  /// Video or Image
  MediaType media_type{0};

  /// ``enum AVPixelFormat`` but using ``int`` so as to avoid including FFmpeg
  /// header.
  int media_format;

  /// The data buffer. Because when using NVDEC decoder, we need to directly
  /// copy the decoded frame from decoder's output buffer, we use a continuous
  /// memory buffer to store the data.
  std::shared_ptr<CUDABuffer2DPitch> buffer;

  ///
  /// True if data is on a CUDA device. Always true for NVDEC.
  bool is_cuda() const;

  ///
  /// Video or Image
  enum MediaType get_media_type() const override;

  NvDecVideoFrames(uint64_t id, MediaType media_type, int media_format);
  NvDecVideoFrames(const NvDecVideoFrames&) = delete;
  NvDecVideoFrames& operator=(const NvDecVideoFrames&) = delete;
  NvDecVideoFrames(NvDecVideoFrames&&) noexcept;
  NvDecVideoFrames& operator=(NvDecVideoFrames&&) noexcept;
  ~NvDecVideoFrames() = default;
#endif
};
} // namespace spdl::core
