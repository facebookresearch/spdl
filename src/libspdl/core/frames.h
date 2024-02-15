#pragma once

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

  virtual std::string get_media_format() const = 0;
  virtual std::string get_media_type() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
struct FFmpegFrames : public DecodedFrames {
  uint64_t id{0};
  MediaType type{0};

  std::vector<AVFrame*> frames{};

  std::string get_media_format() const override;
  std::string get_media_type() const override;

  FFmpegFrames(uint64_t id, MediaType type);
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

  bool is_cuda() const;
  int get_sample_rate() const;
  int get_num_frames() const;
  int get_num_channels() const;
};

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
struct FFmpegVideoFrames : public FFmpegFrames {
  using FFmpegFrames::FFmpegFrames;

  bool is_cuda() const;
  int get_num_frames() const;
  int get_num_planes() const;
  int get_width() const;
  int get_height() const;

  FFmpegVideoFrames slice(int start, int stop, int step) const;
};
} // namespace spdl::core
