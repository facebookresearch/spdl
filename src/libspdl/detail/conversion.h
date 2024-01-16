#pragma once

#include <libspdl/buffers.h>
#include <libspdl/frames.h>

namespace spdl::detail {

Buffer convert_frames(
    const FrameContainer& frames,
    const std::optional<int>& index = std::nullopt);

} // namespace spdl::detail
