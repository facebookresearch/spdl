#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/adoptor/basic.h>
#include <libspdl/core/adoptor/custom.h>
#include <libspdl/core/adoptor/mmap.h>
#include <libspdl/core/decoding.h>
#include <libspdl/core/utils.h>

#include <fmt/core.h>
#include <folly/logging/xlog.h>

using namespace spdl::core;

int main(int argc, char** argv) {
  init_folly(&argc, &argv);

  std::string src =
      "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4";
  std::vector<std::shared_ptr<SourceAdoptor>> adoptors = {
      std::shared_ptr<BasicAdoptor>(new BasicAdoptor()),
      std::shared_ptr<MMapAdoptor>(new MMapAdoptor())};

  XLOG(INFO) << "Executing the coroutine";
  std::vector<MultipleDecodingResult> futures;
  for (auto& adoptor : adoptors) {
    for (int i = 0; i < 10; ++i) {
      std::vector<std::tuple<double, double>> ts{{10 * i, 10 * (i + 1)}};
      futures.emplace_back(decode_video(src, ts, adoptor, {}, {}, ""));
    }
  }

  int i = 0;
  for (auto& future : futures) {
    XLOG(INFO) << "Checking " << ++i;
    try {
      auto frames = future.get();
      for (auto& f : frames) {
        XLOG(INFO) << fmt::format("Decoded: {} frames", f->frames.size());
      }
    } catch (const std::exception& e) {
      XLOG(ERR) << e.what();
    }
  }
  XLOG(INFO) << "Done!";
  return 0;
}
