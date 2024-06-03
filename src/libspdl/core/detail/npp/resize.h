#pragma once

#include <nppi.h>
#include <nvjpeg.h>

namespace spdl::core::detail {

void resize_npp(
    nvjpegOutputFormat_t fmt,
    nvjpegImage_t src,
    int src_width,
    int src_height,
    nvjpegImage_t dst,
    int dst_width,
    int dst_height);

} // namespace spdl::core::detail
