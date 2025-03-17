/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nvjpeg.h>

namespace spdl::cuda::detail {

void resize_npp(
    nvjpegOutputFormat_t fmt,
    nvjpegImage_t src,
    int src_width,
    int src_height,
    nvjpegImage_t dst,
    int dst_width,
    int dst_height);

} // namespace spdl::cuda::detail
