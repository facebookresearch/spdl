#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/frames.h>

namespace spdl::core {

Buffer convert_frames(
    const FrameContainer& frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl::core
