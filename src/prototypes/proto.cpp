#include <fmt/format.h>
#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/BoundedQueue.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/init/Init.h>

#include <libspdl/ffmpeg/utils.h>
#include <libspdl/interface/mmap.h>
#include <libspdl/processors.h>

extern "C" {
#include <libavformat/avformat.h>
}

using namespace spdl;

folly::coro::Task<void> test_stream_avpacket_mmap(
    const std::string_view src,
    const std::vector<double> timestamps,
    FrameQueue& queue,
    const std::optional<OptionDict> options = std::nullopt,
    const std::optional<std::string> format = std::nullopt,
    int buffer_size = 8096) {
  using MMF = spdl::interface::MemoryMappedFile;
  MMF mmf{src.substr(7)};
  auto io_ctx = get_io_ctx(&mmf, buffer_size, MMF::read_packet, MMF::seek);
  auto fmt_ctx = get_input_format_ctx(io_ctx.get(), options, format);
  co_await stream_decode(fmt_ctx.get(), std::move(timestamps), queue);
}

folly::coro::Task<void> test_stream_avpacket(
    const std::string src,
    const std::vector<double> timestamps,
    FrameQueue& queue,
    const std::optional<OptionDict> options = std::nullopt,
    const std::optional<std::string> format = std::nullopt,
    int buffer_size = 8096) {
  if (src.starts_with("mmap://")) {
    co_await test_stream_avpacket_mmap(
        src, timestamps, queue, options, format, buffer_size);
  } else {
    auto fmt_ctx = get_input_format_ctx(src, options, format);
    co_await stream_decode(fmt_ctx.get(), std::move(timestamps), queue);
  }
}

int main(int argc, char** argv) {
  // auto _ = folly::Init{&argc, &argv};
  LOG(INFO) << avcodec_configuration();

  std::vector<std::string> srcs = {
      "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
      "mmap://NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
  };

  auto exec = folly::getGlobalIOExecutor().get();
  FrameQueue queue{100};

  LOG(INFO) << "Executing the coroutine";
  std::vector<folly::SemiFuture<folly::Unit>> sfs;
  int num_jobs = 0;
  for (int i = 0; i < 300; ++i) {
    std::vector<double> ts{5. * i};
    sfs.emplace_back(test_stream_avpacket(srcs[i % srcs.size()], ts, queue)
                         .scheduleOn(exec)
                         .start());
    num_jobs += ts.size();
  }
  folly::coro::blockingWait([&]() -> folly::coro::Task<void> {
    for (int i = 0; i < num_jobs; ++i) {
      auto val = co_await queue.dequeue();
      LOG(INFO) << fmt::format("Dequeue {}: ({} frames)", i, val.frames.size());
    }
  }());

  for (auto& sf : sfs) {
    sf.wait();
  }

  LOG(INFO) << "Done!";
  return 0;
}
