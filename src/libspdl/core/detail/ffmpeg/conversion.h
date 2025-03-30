/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "libspdl/core/detail/ffmpeg/wrappers.h"

namespace spdl::core::detail {

//////////////////////////////////////////////////////////////////////////////////
// Buffer to frame conversion
//////////////////////////////////////////////////////////////////////////////////
AVFramePtr reference_image_buffer(
    AVPixelFormat fmt,
    void* data,
    size_t width,
    size_t height);

} // namespace spdl::core::detail
