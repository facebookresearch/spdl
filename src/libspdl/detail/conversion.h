#pragma once

#include <libspdl/buffers.h>

struct AVFrame;

namespace spdl::detail {

Buffer convert_video_frames(
    const std::vector<AVFrame*>& frames,
    const int plane = -1);

} // namespace spdl::detail
