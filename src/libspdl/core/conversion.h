#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/frames.h>

namespace spdl {

Buffer convert_frames(
    const FrameContainer& frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl
