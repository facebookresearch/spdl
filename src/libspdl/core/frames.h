#pragma once

#include <libspdl/core/types.h>

#include <vector>

struct AVFrame;

namespace spdl::core {

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers

/// Represents series of decoded frames
struct FrameContainer {
  MediaType type;

  std::vector<AVFrame*> frames{};

  FrameContainer(MediaType type);
  // No copy constructors
  FrameContainer(const FrameContainer&) = delete;
  FrameContainer& operator=(const FrameContainer&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  FrameContainer(FrameContainer&&) noexcept = default;
  FrameContainer& operator=(FrameContainer&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~FrameContainer();

  bool is_cuda() const;
  std::string get_format() const;
  int get_num_planes() const;
  int get_width() const;
  int get_height() const;
  int get_sample_rate() const;
  int get_num_samples() const;

  FrameContainer slice(int start, int stop, int step) const;
};

} // namespace spdl::core
