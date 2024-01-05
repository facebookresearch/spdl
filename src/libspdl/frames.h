#pragma once

#include <vector>

struct AVFrame;

namespace spdl {

struct Buffer;

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers

/// Represents series of decoded frames
struct Frames {
  enum class Type { NA, Audio, Video };

  Type type = Type::NA;
  std::vector<AVFrame*> frames{};

  explicit Frames() = default;
  // No copy constructors
  Frames(const Frames&) = delete;
  Frames& operator=(const Frames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  Frames(Frames&&) noexcept = default;
  Frames& operator=(Frames&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~Frames();

  int get_width() const;
  int get_height() const;
  int get_sample_rate() const;

  Frames slice(int start, int stop, int step) const;

  Buffer to_video_buffer(int plane = -1) const;
};

} // namespace spdl
