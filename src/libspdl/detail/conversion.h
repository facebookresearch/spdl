#pragma once

#include <libspdl/buffers.h>
#include <libspdl/frames.h>

namespace spdl::detail {

Buffer convert_video_frames(const Frames& frames, const int plane = -1);

} // namespace spdl::detail
