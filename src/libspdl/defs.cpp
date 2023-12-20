#include <libspdl/defs.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl {

DecodedFrames::~DecodedFrames() {
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

} // namespace spdl
