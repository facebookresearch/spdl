#pragma once

#include <libspdl/frames.h>
#include <libspdl/types.h>
#include <vector>

namespace spdl {

std::vector<Frames> decode(VideoDecodingJob job);

} // namespace spdl
