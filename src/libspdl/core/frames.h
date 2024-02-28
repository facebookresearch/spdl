#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

struct AVFrame;

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Base
////////////////////////////////////////////////////////////////////////////////
struct DecodedFrames {
  virtual ~DecodedFrames() = default;

  virtual enum MediaType get_media_type() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
struct FFmpegFrames : public DecodedFrames {
  uint64_t id{0};

  std::vector<AVFrame*> frames{};

  FFmpegFrames(uint64_t id);
  // No copy constructors
  FFmpegFrames(const FFmpegFrames&) = delete;
  FFmpegFrames& operator=(const FFmpegFrames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  FFmpegFrames(FFmpegFrames&&) noexcept;
  FFmpegFrames& operator=(FFmpegFrames&&) noexcept;
  // Destructor releases AVFrame* resources
  ~FFmpegFrames();
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
struct FFmpegAudioFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  enum MediaType get_media_type() const override;

  bool is_cuda() const;
  int get_sample_rate() const;
  int get_num_frames() const;
  int get_num_channels() const;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Image
////////////////////////////////////////////////////////////////////////////////
struct FFmpegImageFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  enum MediaType get_media_type() const override;

  bool is_cuda() const;
  int get_num_planes() const;
  int get_width() const;
  int get_height() const;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
struct FFmpegVideoFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  enum MediaType get_media_type() const override;

  bool is_cuda() const;
  int get_num_frames() const;
  int get_num_planes() const;
  int get_width() const;
  int get_height() const;

  FFmpegVideoFrames slice(int start, int stop, int step) const;
  FFmpegImageFrames slice(int index) const;
};

#ifdef SPDL_USE_NVDEC
////////////////////////////////////////////////////////////////////////////////
// NVDEC - Video
////////////////////////////////////////////////////////////////////////////////
struct NvDecVideoFrames : public DecodedFrames {
  uint64_t id{0};
  int device_index;
  MediaType media_type{0};

  // enum AVPixelFormat but using int so as not to include FFmpeg header
  int media_format;

  std::shared_ptr<CUDABuffer2DPitch> buffer;

  bool is_cuda() const;
  enum MediaType get_media_type() const override;

  NvDecVideoFrames(
      uint64_t id,
      int device_index,
      MediaType media_type,
      int media_format);
  NvDecVideoFrames(const NvDecVideoFrames&) = delete;
  NvDecVideoFrames& operator=(const NvDecVideoFrames&) = delete;
  NvDecVideoFrames(NvDecVideoFrames&&) noexcept;
  NvDecVideoFrames& operator=(NvDecVideoFrames&&) noexcept;
  // Destructor releases AVFrame* resources
  ~NvDecVideoFrames() = default;
};
#endif
} // namespace spdl::core
