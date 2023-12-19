#include <libspdl/buffer.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl {

DecodedVideoFrames::~DecodedVideoFrames() {
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

} // namespace spdl
