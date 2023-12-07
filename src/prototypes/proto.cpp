#include <fmt/format.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/init/Init.h>

#include <libspdl/processors.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

int main(int argc, char** argv) {
  // auto _ = folly::Init{&argc, &argv};
  LOG(INFO) << avcodec_configuration();

  std::vector<std::string> srcs = {
      "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
      "mmap://NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
  };

  spdl::Engine engine{64, 3, 100};

  LOG(INFO) << "Executing the coroutine";
  int num_jobs = 0;
  for (int i = 0; i < 300; ++i) {
    std::vector<double> ts{5. * i};
    engine.enqueue({srcs[i % srcs.size()], ts});
    num_jobs += ts.size();
  }
  folly::coro::blockingWait([&]() -> folly::coro::Task<void> {
    for (int i = 0; i < num_jobs; ++i) {
      auto val = co_await engine.frame_queue.dequeue();
      LOG(INFO) << fmt::format("Dequeue {}: ({} frames)", i, val.frames.size());
    }
  }());

  for (auto& sf : engine.sfs) {
    sf.wait();
  }

  LOG(INFO) << "Done!";
  return 0;
}
