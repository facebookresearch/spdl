#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/adoptor/basic.h>
#include <libspdl/core/adoptor/custom.h>
#include <libspdl/core/adoptor/mmap.h>
#include <libspdl/core/decoding.h>
#include <libspdl/core/utils.h>

#include <folly/logging/xlog.h>

using namespace spdl::core;

int main(int argc, char** argv) {
  init_folly(&argc, &argv);

  std::string src =
      "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4";
  std::vector<std::shared_ptr<SourceAdoptor>> adoptors = {
      std::shared_ptr<BasicAdoptor>(new BasicAdoptor("/home/moto/")),
      std::shared_ptr<MMapAdoptor>(new MMapAdoptor("/home/moto/"))};

  XLOG(INFO) << "Executing the coroutine";
  for (auto& adoptor : adoptors) {
    std::vector<std::tuple<double, double>> ts{{5., 10.}};
    try {
      auto frames = decode_video(src, ts, adoptor, {}, {}, "");
    } catch (const std::exception& e) {
      XLOG(ERR) << e.what();
    }
  }
  XLOG(INFO) << "Done!";
  return 0;
}
