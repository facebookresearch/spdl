/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 2, 100)
#define GET_CHANNEL(x) x->ch_layout.nb_channels
#else
#define GET_CHANNEL(x) x->channels
#endif

namespace {
template <size_t depth, ElemClass type, bool is_planar>
CPUBufferPtr convert_frames(
    const std::vector<const FFmpegAudioFrames*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  size_t num_frames = batch.at(0)->get_num_frames();
  size_t num_channels = GET_CHANNEL(batch.at(0)->get_frames().at(0));

  if constexpr (is_planar) {
    auto buf = cpu_buffer(
        {batch.size(), num_channels, num_frames},
        type,
        depth,
        std::move(storage));
    uint8_t* dst = static_cast<uint8_t*>(buf->data());
    for (auto frames_ptr : batch) {
      auto fs = frames_ptr->get_frames();
      for (int i = 0; i < num_channels; ++i) {
        for (auto frame : fs) {
          int plane_size = depth * frame->nb_samples;
          memcpy(dst, frame->extended_data[i], plane_size);
          dst += plane_size;
        }
      }
    }
    return buf;
  } else {
    auto buf = cpu_buffer(
        {batch.size(), num_frames, num_channels},
        type,
        depth,
        std::move(storage));
    uint8_t* dst = static_cast<uint8_t*>(buf->data());
    for (auto frames_ptr : batch) {
      auto fs = frames_ptr->get_frames();
      for (auto frame : fs) {
        int plane_size = depth * frame->nb_samples * num_channels;
        memcpy(dst, frame->extended_data[0], plane_size);
        dst += plane_size;
      }
    }
    return buf;
  }
}
} // namespace

template <>
CPUBufferPtr convert_frames(
    const std::vector<const FFmpegAudioFrames*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  if (batch.empty()) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto frames_ptr0 = batch.at(0);
  auto num_frames0 = frames_ptr0->get_num_frames();
  if (num_frames0 == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto frames0 = frames_ptr0->get_frames();
  auto num_channels0 = GET_CHANNEL(frames0.at(0));
  auto sample_fmt0 = static_cast<AVSampleFormat>(frames0.at(0)->format);
  for (auto& frames_ptr : batch) {
    auto num_frames = frames_ptr->get_num_frames();
    if (num_frames == 0) {
      SPDL_FAIL("No frame to convert to buffer.");
    }
    auto frames = frames_ptr->get_frames();
    auto sample_fmt = static_cast<AVSampleFormat>(frames.at(0)->format);
    auto num_channels = GET_CHANNEL(frames.at(0));
    if (num_frames == 0) {
      SPDL_FAIL("No frame to convert to buffer.");
    }
    if (num_frames != num_frames0) {
      SPDL_FAIL(fmt::format(
          "Inconsistent number of frames: {} vs {}", num_frames0, num_frames));
    }
    if (sample_fmt0 != sample_fmt) {
      SPDL_FAIL(fmt::format(
          "Inconsistent sample format: {} vs {}",
          av_get_sample_fmt_name(sample_fmt0),
          av_get_sample_fmt_name(sample_fmt)));
    }
    if (num_channels0 != num_channels) {
      SPDL_FAIL(fmt::format(
          "Inconsistent number of channels: {} vs {}",
          num_channels0,
          num_channels));
    }
  }
  switch (sample_fmt0) {
    case AV_SAMPLE_FMT_U8:
      return convert_frames<1, ElemClass::UInt, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_U8P:
      return convert_frames<1, ElemClass::UInt, true>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_S16:
      return convert_frames<2, ElemClass::Int, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_S16P:
      return convert_frames<2, ElemClass::Int, true>(batch, std::move(storage));
    case AV_SAMPLE_FMT_S32:
      return convert_frames<4, ElemClass::Int, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_S32P:
      return convert_frames<4, ElemClass::Int, true>(batch, std::move(storage));
    case AV_SAMPLE_FMT_FLT:
      return convert_frames<4, ElemClass::Float, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_FLTP:
      return convert_frames<4, ElemClass::Float, true>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_S64:
      return convert_frames<8, ElemClass::Int, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_S64P:
      return convert_frames<8, ElemClass::Int, true>(batch, std::move(storage));
    case AV_SAMPLE_FMT_DBL:
      return convert_frames<8, ElemClass::Float, false>(
          batch, std::move(storage));
    case AV_SAMPLE_FMT_DBLP:
      return convert_frames<8, ElemClass::Float, true>(
          batch, std::move(storage));
    default:
      SPDL_FAIL_INTERNAL(fmt::format(
          "Unexpected sample format: {}", av_get_sample_fmt_name(sample_fmt0)));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Image / Video
////////////////////////////////////////////////////////////////////////////////
namespace {
void copy_2d(
    uint8_t* src,
    int height,
    int width,
    int src_linesize,
    uint8_t** dst,
    int dst_linesize,
    int depth = 1) {
  TRACE_EVENT("decoding", "conversion::copy_2d");
  for (int h = 0; h < height; ++h) {
    memcpy(*dst, src, (size_t)width * depth);
    src += src_linesize;
    *dst += dst_linesize;
  }
}

void copy_interleaved(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    unsigned int num_channels,
    size_t w,
    size_t h,
    int depth = 1) {
  size_t wc = num_channels * w * depth;
  for (const auto& f : frames) {
    copy_2d(f->data[0], f->height, wc, f->linesize[0], &dst, wc, depth);
  }
}

template <MediaType media_type>
CPUBufferPtr convert_interleaved(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    size_t num_channels,
    std::shared_ptr<CPUStorage> storage) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  auto buf = cpu_buffer(
      {batch.size(), num_frames, h, w, num_channels},
      ElemClass::UInt,
      sizeof(uint8_t),
      std::move(storage));
  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_interleaved(frames->get_frames(), dst, num_channels, w, h);
    dst += num_frames * h * w * num_channels;
  }
  return buf;
}

void copy_planer(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    size_t num_planes,
    size_t w,
    size_t h,
    int depth) {
  for (const auto& f : frames) {
    for (size_t c = 0; c < num_planes; ++c) {
      copy_2d(f->data[c], h, w, f->linesize[c], &dst, w * depth, depth);
    }
  }
}

template <MediaType media_type>
CPUBufferPtr convert_planer(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    size_t num_planes,
    int depth,
    std::shared_ptr<CPUStorage> storage) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  auto buf = cpu_buffer(
      {batch.size(), num_frames, num_planes, h, w},
      ElemClass::UInt,
      depth,
      std::move(storage));
  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_planer(frames->get_frames(), dst, num_planes, w, h, depth);
    dst += num_frames * num_planes * h * w * depth;
  }
  return buf;
}

void copy_yuv420p(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    size_t w,
    size_t h) {
  size_t h2 = h / 2, w2 = w / 2;
  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    uint8_t* dst2 = dst + w2;
    copy_2d(f->data[1], h2, w2, f->linesize[1], &dst, w);
    copy_2d(f->data[2], h2, w2, f->linesize[2], &dst2, w);
  }
}

template <MediaType media_type>
CPUBufferPtr convert_yuv420p(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h2, w},
      ElemClass::UInt,
      sizeof(uint8_t),
      std::move(storage));
  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_yuv420p(frames->get_frames(), dst, w, h);
    dst += num_frames * (h + h2) * w;
  }
  return buf;
}

void copy_yuv422p(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    size_t w,
    size_t h) {
  assert(w % 2 == 0);
  size_t w2 = w / 2;

  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    uint8_t* dst2 = dst + w2;
    copy_2d(f->data[1], h, w2, f->linesize[1], &dst, w);
    copy_2d(f->data[2], h, w2, f->linesize[2], &dst2, w);
  }
}

template <MediaType media_type>
CPUBufferPtr convert_yuv422p(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  assert(w % 2 == 0);

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h, w},
      ElemClass::UInt,
      sizeof(uint8_t),
      std::move(storage));

  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_yuv422p(frames->get_frames(), dst, w, h);
    dst += num_frames * (h + h) * w;
  }
  return buf;
}

void copy_nv12(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    size_t w,
    size_t h) {
  size_t h2 = h / 2;
  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    copy_2d(f->data[1], h2, w, f->linesize[1], &dst, w);
  }
}

template <MediaType media_type>
CPUBufferPtr convert_nv12(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h2, w},
      ElemClass::UInt,
      sizeof(uint8_t),
      std::move(storage));
  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_nv12(frames->get_frames(), dst, w, h);
    dst += num_frames * (h + h2) * w;
  }
  return buf;
}
} // namespace

namespace {
template <MediaType media_type>
void check_frame_consistency(const FFmpegFrames<media_type>* frames_ptr)
  requires(media_type != MediaType::Audio)
{
  auto numel = frames_ptr->get_num_frames();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  if constexpr (media_type == MediaType::Image) {
    if (numel != 1) {
      SPDL_FAIL_INTERNAL(fmt::format(
          "There must be exactly one frame to convert to buffer. Found: {}",
          numel));
    }
  }
  auto frames = frames_ptr->get_frames();
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  if (pix_fmt == AV_PIX_FMT_CUDA) {
    SPDL_FAIL_INTERNAL("FFmpeg-native CUDA frames are not supported.");
  }

  int height = frames[0]->height, width = frames[0]->width;
  for (auto* f : frames) {
    if (f->height != height || f->width != width) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same size. "
          "Reference WxH = {}x{}, found {}x{}.",
          height,
          width,
          f->height,
          f->width));
    }
    if (static_cast<AVPixelFormat>(f->format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
  }
}

template <MediaType media_type>
void check_batch_frame_consistency(
    const std::vector<const FFmpegFrames<media_type>*>& batch) {
  if (batch.empty()) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto& frames0 = batch.at(0)->get_frames();
  auto w = frames0.at(0)->width, h = frames0.at(0)->height;
  auto num_frames = frames0.size();

  auto pix_fmt0 = static_cast<AVPixelFormat>(frames0.at(0)->format);
  for (auto& frames_ptr : batch) {
    check_frame_consistency(frames_ptr);
    auto frames = frames_ptr->get_frames();
    auto pix_fmt = static_cast<AVPixelFormat>(frames.at(0)->format);
    if (pix_fmt != pix_fmt0) {
      SPDL_FAIL(fmt::format(
          "The input video frames must have the same pixel format. Expected {}, but found {}",
          av_get_pix_fmt_name(pix_fmt0),
          av_get_pix_fmt_name(pix_fmt)));
    }
    if (frames.size() != num_frames) {
      SPDL_FAIL(fmt::format(
          "The number of frames must be the same. Expected {}, but found {}",
          num_frames,
          frames.size()));
    }
    for (auto& frame : frames) {
      if (frame->width != w || frame->height != h) {
        SPDL_FAIL(fmt::format(
            "The input video frames must be the same size. Expected {}x{}, but found {}x{}",
            w,
            h,
            frame->width,
            frame->height));
      }
    }
  }
}
} // namespace

// Note:
//
// YUV is a limited range (16 - 235), while YUVJ is a full range. (0-255).
//
// YUVJ == YUV + AVCOL_RANGE_JPEG
//
// AVCOL_RANGE_JPEG has slight different value range for Chroma.
// (1 - 255 instead of 0 - 255)
// https://ffmpeg.org/doxygen/5.1/pixfmt_8h.html#a3da0bf691418bc22c4bcbe6583ad589a
//
// FFmpeg emits a warning like
// `deprecated pixel format used, make sure you did set range correctly`
//
// See also: https://superuser.com/a/1273941
//
// It might be more appropriate to convert the limited range to the full range
// for YUV, but for now, it copies data as-is for both YUV and YUVJ.
template <MediaType media_type>
CPUBufferPtr convert_frames(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  TRACE_EVENT("decoding", "core::convert_frames");
  check_batch_frame_consistency(batch);

  auto ret = [&]() {
    auto& ref_frames = batch.at(0)->get_frames();
    auto pix_fmt = static_cast<AVPixelFormat>(ref_frames.at(0)->format);
    switch (pix_fmt) {
      case AV_PIX_FMT_GRAY8:
        // Technically, not a planer format, but it's the same.
        return convert_planer(batch, 1, sizeof(uint8_t), std::move(storage));
      case AV_PIX_FMT_GRAY16BE:
        return convert_planer(batch, 1, sizeof(uint16_t), std::move(storage));
      case AV_PIX_FMT_RGBA:
        return convert_interleaved(batch, 4, std::move(storage));
      case AV_PIX_FMT_RGB24:
        return convert_interleaved(batch, 3, std::move(storage));
      case AV_PIX_FMT_YUVJ444P:
      case AV_PIX_FMT_YUV444P:
        return convert_planer(batch, 3, sizeof(uint8_t), std::move(storage));
      case AV_PIX_FMT_YUVJ420P:
      case AV_PIX_FMT_YUV420P:
        return convert_yuv420p(batch, std::move(storage));
      case AV_PIX_FMT_YUVJ422P:
      case AV_PIX_FMT_YUV422P:
        return convert_yuv422p(batch, std::move(storage));
      case AV_PIX_FMT_NV12: {
        return convert_nv12(batch, std::move(storage));
      }
      default:
        SPDL_FAIL(fmt::format(
            "Unsupported pixel format: {}", av_get_pix_fmt_name(pix_fmt)));
    }
  }();

  if constexpr (media_type == MediaType::Image) {
    // Remove the 2nd num_frame dimension (which is 1).
    // BNCHW -> BCHW.
    ret->shape.erase(std::next(ret->shape.begin()));
  }

  return ret;
}

template CPUBufferPtr convert_frames(
    const std::vector<const FFmpegImageFrames*>& batch,
    std::shared_ptr<CPUStorage> storage);

template CPUBufferPtr convert_frames(
    const std::vector<const FFmpegVideoFrames*>& batch,
    std::shared_ptr<CPUStorage> storage);

namespace detail {
////////////////////////////////////////////////////////////////////////////////
// Buffer to frame
////////////////////////////////////////////////////////////////////////////////
namespace {
AVFrameViewPtr get_video_frame(AVPixelFormat fmt, size_t width, size_t height) {
  AVFrameViewPtr ret{CHECK_AVALLOCATE(av_frame_alloc())};
  ret->format = fmt;
  ret->width = width;
  ret->height = height;
  ret->pts = 0;
  return ret;
}

void ref_interweaved(
    AVFrame* frame,
    void* data,
    int num_channels,
    int bit_depth = 1) {
  frame->data[0] = reinterpret_cast<uint8_t*>(data);
  frame->linesize[0] = frame->width * num_channels * bit_depth;
}

void ref_planar(AVFrame* frame, void* data, int num_channels) {
  auto src = reinterpret_cast<uint8_t*>(data);
  for (int c = 0; c < num_channels; ++c) {
    frame->data[c] = src;
    frame->linesize[c] = frame->width;
    src += frame->height * frame->width;
  }
}
} // namespace

AVFrameViewPtr reference_image_buffer(
    AVPixelFormat fmt,
    void* data,
    size_t width,
    size_t height) {
  auto frame = get_video_frame(fmt, width, height);
  switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      ref_interweaved(frame.get(), data, 3);
      break;
    case AV_PIX_FMT_GRAY8:
      ref_interweaved(frame.get(), data, 1);
      break;
    case AV_PIX_FMT_GRAY16BE:
      ref_interweaved(frame.get(), data, 1, 2);
      break;
    case AV_PIX_FMT_YUV444P:
      ref_planar(frame.get(), data, 3);
      break;
    default:
      SPDL_FAIL(fmt::format(
          "Unsupported source pixel format: {}", av_get_pix_fmt_name(fmt)));
  }
  return frame;
}

} // namespace detail
} // namespace spdl::core
