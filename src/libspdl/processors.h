#pragma once

#include <optional>
#include <string>
#include <vector>

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BoundedQueue.h>
#include <folly/experimental/coro/Task.h>

#include <libspdl/common.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

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
  AVRational time_base{0, 1};
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
  using Executor = folly::CPUThreadPoolExecutor;

 public:
  struct Job {
    std::string path;
    std::vector<double> timestamps;
    std::optional<std::string> format = std::nullopt;
    std::optional<OptionDict> format_options = std::nullopt;
    int buffer_size = 8096;
    std::optional<std::string> decoder = std::nullopt;
    std::optional<OptionDict> decoder_options = std::nullopt;
    int cuda_device_index = -1;
    std::optional<AVRational> frame_rate = std::nullopt;
    std::optional<int> width = std::nullopt;
    std::optional<int> height = std::nullopt;
    std::optional<std::string> pix_fmt = std::nullopt;
  };

 private:
  std::unique_ptr<Executor> io_task_executors;
  std::unique_ptr<Executor> decoding_task_executors;

  folly::Executor::KeepAlive<> io_exec;
  folly::Executor::KeepAlive<> decoding_exec;

 public:
  // temporarily public until we figure out a good way to do bookkeeping
  FrameQueue frame_queue;
  std::vector<folly::SemiFuture<folly::Unit>> sfs;

  Engine(
      size_t num_io_threads,
      size_t num_decoding_threads,
      size_t frame_queue_size);

  void enqueue(Job job);

  VideoBuffer dequeue();
};

} // namespace spdl
