#pragma once

#include <libspdl/types.h>
#include <vector>

struct AVFrame;

namespace spdl {

struct Buffer;

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers

/// Represents series of decoded frames
struct Frames {
  MediaType type = MediaType::NA;

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

  bool is_cuda() const;
  std::string get_format() const;
  int get_num_planes() const;
  int get_width() const;
  int get_height() const;
  int get_sample_rate() const;
  int get_num_samples() const;

  Frames slice(int start, int stop, int step) const;

  Buffer to_buffer(const std::optional<int>& index = std::nullopt) const;
};

} // namespace spdl
