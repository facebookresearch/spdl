/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

struct AVCodecContext;

namespace spdl::core {
class FrameArena;
}

namespace spdl::core::detail {
void install_arena(AVCodecContext* ctx, FrameArena* arena);
} // namespace spdl::core::detail
