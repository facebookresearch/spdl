#pragma once

#include <vector>

struct AVFrame;

namespace spdl {

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers

/// Represents series of decoded frames
struct Frames {
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
};

Frames slice_frames(const Frames& orig, int start, int stop, int step);

} // namespace spdl
