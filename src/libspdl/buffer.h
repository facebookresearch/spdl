#pragma once

#include <vector>

struct AVFrame;

namespace spdl {

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
// Each AVFrame* is expected to be created by av_read_frame.
// Therefore, they are reference counted and the counter should be 1.
// When destructing, they will be first unreferenced with av_frame_unref,
// then the data must be released with av_frame_free.
struct DecodedVideoFrames {
  std::vector<AVFrame*> frames{};

  explicit DecodedVideoFrames() = default;
  // No copy constructors
  DecodedVideoFrames(const DecodedVideoFrames&) = delete;
  DecodedVideoFrames& operator=(const DecodedVideoFrames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  DecodedVideoFrames(DecodedVideoFrames&&) noexcept = default;
  DecodedVideoFrames& operator=(DecodedVideoFrames&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~DecodedVideoFrames();
};

} // namespace spdl
