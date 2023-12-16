#pragma once

#include <optional>
#include <string>
#include <vector>

#include <folly/experimental/coro/BoundedQueue.h>

#include <libspdl/defs.h>

struct AVFrame;

namespace spdl {

// Temporaly output struct
// until we have a proper way to expose the output of Engine to client code.

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
// Each AVFrame* is expected to be created by av_read_frame.
// Therefore, they are reference counted and the counter should be 1.
// When destructing, they will be first unreferenced with av_frame_unref,
// then the data must be released with av_frame_free.
struct PackagedAVFrames {
  Rational time_base{0, 1};
  std::vector<AVFrame*> frames{};

  explicit PackagedAVFrames() = default;
  // No copy constructors
  PackagedAVFrames(const PackagedAVFrames&) = delete;
  PackagedAVFrames& operator=(const PackagedAVFrames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  PackagedAVFrames(PackagedAVFrames&&) noexcept = default;
  PackagedAVFrames& operator=(PackagedAVFrames&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~PackagedAVFrames();
};

// Temp
using FrameQueue = folly::coro::BoundedQueue<PackagedAVFrames, false, true>;

// buffer class to be exposed to python
struct VideoBuffer {
  size_t n, c, h, w;
  bool channel_last = false;
  std::vector<uint8_t> data;
};

//////////////////////////////////////////////////////////////////////////////
// Engine
//////////////////////////////////////////////////////////////////////////////
class Engine {
 public:
  // temporarily public until we figure out a good way to do bookkeeping
  FrameQueue frame_queue;
  std::vector<folly::SemiFuture<folly::Unit>> sfs;

  Engine(size_t frame_queue_size);

  void enqueue(VideoDecodingJob job);

  VideoBuffer dequeue();
};

} // namespace spdl
