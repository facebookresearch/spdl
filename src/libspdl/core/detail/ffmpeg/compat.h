/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

extern "C" {
#include <libavutil/version.h>
}

// 6 ~
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 17, 100)
#define GET_NUM_CHANNELS(x) x->ch_layout.nb_channels
#define GET_LAYOUT(x) x->ch_layout.u.mask
#else
#define GET_NUM_CHANNELS(x) x->channels
#define GET_LAYOUT(x) x->channel_layout
#endif
