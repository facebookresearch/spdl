#include <fmt/format.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <libspdl/processors.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

int main(int argc, char** argv) {
  // auto _ = folly::Init{&argc, &argv};
  XLOG(INFO) << avcodec_configuration();

  std::vector<std::string> srcs = {
      "/home/moto/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
      "mmap:///home/moto/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
  };

  spdl::Engine engine{64, 32, 100};

  XLOG(INFO) << "Executing the coroutine";
  int num_jobs = 0;
  for (int i = 0; i < 300; ++i) {
    std::vector<double> ts{5. * i};
    engine.enqueue({srcs[i % srcs.size()], ts});
    num_jobs += ts.size();
  }
  folly::coro::blockingWait([&]() -> folly::coro::Task<void> {
    for (int i = 0; i < num_jobs; ++i) {
      auto val = co_await engine.frame_queue.dequeue();
      XLOG(INFO) << fmt::format(
          "Dequeue {}: ({} frames)", i, val.frames.size());
    }
  }());

  for (auto& sf : engine.sfs) {
    sf.wait();
  }

  XLOG(INFO) << "Done!";
  return 0;
}
