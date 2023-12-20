#pragma once

#include <optional>
#include <string>
#include <vector>

#include <folly/experimental/coro/BoundedQueue.h>

#include <libspdl/frames.h>

struct AVFrame;

namespace spdl {

// Temp
using FrameQueue = folly::coro::BoundedQueue<Frames, false, true>;

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

  Frames dequeue();
};

} // namespace spdl
