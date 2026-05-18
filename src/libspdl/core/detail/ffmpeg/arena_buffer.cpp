/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/arena_buffer.h"

#include <libspdl/core/frame_arena.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
}

#include "libspdl/core/detail/ffmpeg/compat.h"

namespace spdl::core::detail {
namespace {

constexpr int kAlign = 32;

void arena_free_callback(void* opaque, uint8_t* data) {
  auto* arena = static_cast<FrameArena*>(opaque);
  arena->deallocate(static_cast<void*>(data));
}

int arena_get_buffer2(AVCodecContext* ctx, AVFrame* frame, int /*flags*/) {
  auto* arena = static_cast<FrameArena*>(ctx->opaque);

  int size;
  if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
    size = av_image_get_buffer_size(
        static_cast<AVPixelFormat>(frame->format),
        frame->width,
        frame->height,
        kAlign);
  } else {
    int channels = GET_NUM_CHANNELS(frame);
    size = av_samples_get_buffer_size(
        nullptr,
        channels,
        frame->nb_samples,
        static_cast<AVSampleFormat>(frame->format),
        kAlign);
  }

  if (size < 0) {
    return size;
  }

  auto* buf = static_cast<uint8_t*>(arena->allocate(size));
  if (!buf) {
    return AVERROR(ENOMEM);
  }

  frame->buf[0] = av_buffer_create(buf, size, arena_free_callback, arena, 0);
  if (!frame->buf[0]) {
    arena->deallocate(buf);
    return AVERROR(ENOMEM);
  }

  if (ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
    av_image_fill_arrays(
        frame->data,
        frame->linesize,
        buf,
        static_cast<AVPixelFormat>(frame->format),
        frame->width,
        frame->height,
        kAlign);
  } else {
    av_samples_fill_arrays(
        frame->data,
        frame->linesize,
        buf,
        GET_NUM_CHANNELS(frame),
        frame->nb_samples,
        static_cast<AVSampleFormat>(frame->format),
        kAlign);
  }

  return 0;
}

} // namespace

void install_arena(AVCodecContext* ctx, FrameArena* arena) {
  ctx->opaque = arena;
  ctx->get_buffer2 = arena_get_buffer2;
}

} // namespace spdl::core::detail
