/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

extern "C" {
#include <libavformat/version.h>
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

// https://github.com/FFmpeg/FFmpeg/blob/4e6debe1df7d53f3f59b37449b82265d5c08a172/doc/APIchanges#L252-L260
// Starting from libavformat 59 (ffmpeg 5),
// AVInputFormat is const and related functions expect constant.
#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVFORMAT_CONST const
#else
#define AVFORMAT_CONST
#endif
