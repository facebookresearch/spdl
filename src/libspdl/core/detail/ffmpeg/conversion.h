#pragma once
#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

//////////////////////////////////////////////////////////////////////////////////
// Buffer to frame conversion
//////////////////////////////////////////////////////////////////////////////////
AVFrameViewPtr reference_image_buffer(
    AVPixelFormat fmt,
    void* data,
    size_t width,
    size_t height);

} // namespace spdl::core::detail
