#pragma once

#include <vector>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BoundedQueue.h>
#include <folly/experimental/coro/Task.h>

struct AVFrame;
struct AVFormatContext;

namespace spdl {

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
// Each AVFrame* is expected to be created by av_read_frame.
// Therefore, they are reference counted and the counter should be 1.
// When destructing, they will be first unreferenced with av_frame_unref,
// then the data must be released with av_frame_free.
struct PackagedAVFrames {
  std::vector<AVFrame*> frames{};

  PackagedAVFrames() = default;
  // No copy constructors
  PackagedAVFrames(const PackagedAVFrames&) = delete;
  PackagedAVFrames& operator=(const PackagedAVFrames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  PackagedAVFrames(PackagedAVFrames&&) noexcept = default;
  PackagedAVFrames& operator=(PackagedAVFrames&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~PackagedAVFrames();
};

using FrameQueue = folly::coro::BoundedQueue<PackagedAVFrames, false, true>;

//////////////////////////////////////////////////////////////////////////////
// Processor
//////////////////////////////////////////////////////////////////////////////
folly::coro::Task<void> stream_decode(
    AVFormatContext* fmt_ctx,
    const std::vector<double> timestamps,
    FrameQueue& queue);

} // namespace spdl
