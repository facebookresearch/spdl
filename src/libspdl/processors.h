#pragma once

#include <optional>
#include <string>
#include <vector>

#include <folly/experimental/coro/BoundedQueue.h>

#include <libspdl/defs.h>
#include <libspdl/buffer.h>

struct AVFrame;

namespace spdl {

// Temp
using FrameQueue = folly::coro::BoundedQueue<DecodedVideoFrames, false, true>;

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
