/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
namespace {
template <size_t depth, ElemClass type, bool is_planar>
CPUBufferPtr convert_frames(
    const std::vector<const AudioFrames*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  size_t num_frames = batch.at(0)->get_num_frames();
  size_t num_channels = GET_NUM_CHANNELS(batch.at(0)->get_frames().at(0));

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
    const std::vector<const AudioFrames*>& batch,
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
  auto num_channels0 = GET_NUM_CHANNELS(frames0.at(0));
  auto sample_fmt0 = static_cast<AVSampleFormat>(frames0.at(0)->format);
  for (auto& frames_ptr : batch) {
    auto num_frames = frames_ptr->get_num_frames();
    if (num_frames == 0) {
      SPDL_FAIL("No frame to convert to buffer.");
    }
    auto frames = frames_ptr->get_frames();
    auto sample_fmt = static_cast<AVSampleFormat>(frames.at(0)->format);
    auto num_channels = GET_NUM_CHANNELS(frames.at(0));
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
template <MediaType media>
void copy(
    AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    CPUBuffer* buf) {
  auto dst = (uint8_t*)buf->data();
  auto dst_size = buf->shape[2] * buf->shape[3] * buf->shape[4] * buf->depth;
  for (auto& frames : batch) {
    for (const auto& f : frames->get_frames()) {
      CHECK_AVERROR(
          av_image_copy_to_buffer(
              dst,
              dst_size,
              f->data,
              f->linesize,
              pix_fmt,
              f->width,
              f->height,
              buf->depth),
          "Failed to copy image data.");
      dst += dst_size;
    }
  }
}

template <MediaType media>
CPUBufferPtr convert_interleaved(
    enum AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage,
    int depth = 1) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  int num_channels = av_pix_fmt_desc_get(pix_fmt)->nb_components;
  assert(num_channels > 0);

  auto buf = cpu_buffer(
      {batch.size(), num_frames, h, w, (unsigned int)num_channels},
      ElemClass::UInt,
      depth,
      std::move(storage));
  copy(pix_fmt, batch, buf.get());
  return buf;
}

template <MediaType media>
CPUBufferPtr convert_planer(
    enum AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage,
    int depth = 1) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  int num_planes = av_pix_fmt_count_planes(pix_fmt);
  if (num_planes <= 0) {
    SPDL_FAIL("Failed to fetch the number of planes.");
  }

  auto buf = cpu_buffer(
      {batch.size(), num_frames, (unsigned int)num_planes, h, w},
      ElemClass::UInt,
      depth,
      std::move(storage));
  copy(pix_fmt, batch, buf.get());
  return buf;
}

template <MediaType media>
CPUBufferPtr convert_yuv420p(
    enum AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage,
    int depth = 1) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h2, w},
      ElemClass::UInt,
      depth,
      std::move(storage));
  copy(pix_fmt, batch, buf.get());
  return buf;
}

template <MediaType media>
CPUBufferPtr convert_yuv422p(
    enum AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage,
    int depth = 1) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h, w},
      ElemClass::UInt,
      depth,
      std::move(storage));
  copy(pix_fmt, batch, buf.get());
  return buf;
}

template <MediaType media>
CPUBufferPtr convert_nv12(
    enum AVPixelFormat pix_fmt,
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage,
    int depth = 1) {
  auto& ref_frames = batch.at(0)->get_frames();
  size_t w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
  auto num_frames = ref_frames.size();

  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer(
      {batch.size(), num_frames, 1, h + h2, w},
      ElemClass::UInt,
      depth,
      std::move(storage));
  copy(pix_fmt, batch, buf.get());
  return buf;
}
} // namespace

namespace {
template <MediaType media>
void check_frame_consistency(const Frames<media>* frames_ptr)
  requires(media != MediaType::Audio)
{
  auto numel = frames_ptr->get_num_frames();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  if constexpr (media == MediaType::Image) {
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

template <MediaType media>
void check_batch_frame_consistency(
    const std::vector<const Frames<media>*>& batch) {
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
template <MediaType media>
CPUBufferPtr convert_frames(
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage) {
  TRACE_EVENT("decoding", "core::convert_frames");
  check_batch_frame_consistency(batch);

  auto ret = [&]() {
    auto& ref_frames = batch.at(0)->get_frames();
    auto pix_fmt = static_cast<AVPixelFormat>(ref_frames.at(0)->format);
    switch (pix_fmt) {
      case AV_PIX_FMT_GRAY8:
      case AV_PIX_FMT_RGBA:
      case AV_PIX_FMT_RGB24:
        return convert_interleaved(pix_fmt, batch, std::move(storage));
      case AV_PIX_FMT_GRAY16BE:
      case AV_PIX_FMT_GRAY16LE:
        return convert_planer(pix_fmt, batch, std::move(storage), 2);
      case AV_PIX_FMT_YUVJ444P:
      case AV_PIX_FMT_YUV444P:
        return convert_planer(pix_fmt, batch, std::move(storage));
      case AV_PIX_FMT_YUVJ420P:
      case AV_PIX_FMT_YUV420P:
        return convert_yuv420p(pix_fmt, batch, std::move(storage));
      case AV_PIX_FMT_YUVJ422P:
      case AV_PIX_FMT_YUV422P:
        return convert_yuv422p(pix_fmt, batch, std::move(storage));
      case AV_PIX_FMT_NV12: {
        return convert_nv12(pix_fmt, batch, std::move(storage));
      }
      default:
        SPDL_FAIL(fmt::format(
            "Unsupported pixel format: {}", av_get_pix_fmt_name(pix_fmt)));
    }
  }();

  if constexpr (media == MediaType::Image) {
    // Remove the 2nd num_frame dimension (which is 1).
    // BNCHW -> BCHW.
    ret->shape.erase(std::next(ret->shape.begin()));
  }

  return ret;
}

template CPUBufferPtr convert_frames(
    const std::vector<const ImageFrames*>& batch,
    std::shared_ptr<CPUStorage> storage);

template CPUBufferPtr convert_frames(
    const std::vector<const VideoFrames*>& batch,
    std::shared_ptr<CPUStorage> storage);

namespace detail {
////////////////////////////////////////////////////////////////////////////////
// Buffer to frame
////////////////////////////////////////////////////////////////////////////////
namespace {
AVFramePtr
get_video_frame(AVPixelFormat fmt, size_t width, size_t height, int64_t pts) {
  AVFramePtr ret{CHECK_AVALLOCATE(av_frame_alloc())};
  ret->format = fmt;
  ret->width = width;
  ret->height = height;
  ret->pts = pts;
  return ret;
}

void no_free(void* opaque, uint8_t* data) {
  VLOG(15) << fmt::format("Not free-ing data @ {}", (void*)data);
}

// Create an AVBufferRef object that points to an existing buffer,
// but not owning it.
AVBufferRef* create_reference_buffer(uint8_t* data, int size) {
  return av_buffer_create(
      data, size, no_free, nullptr, AV_BUFFER_FLAG_READONLY);
}

void ref_interweaved(
    AVFrame* frame,
    void* data,
    int num_channels,
    int bit_depth = 1) {
  auto* src = static_cast<uint8_t*>(data);
  int linesize = frame->width * num_channels * bit_depth;
  frame->data[0] = src;
  frame->linesize[0] = linesize;
  frame->buf[0] = create_reference_buffer(src, linesize * frame->height);
}

void ref_planar(AVFrame* frame, void* data, int num_channels) {
  auto src = static_cast<uint8_t*>(data);
  auto size = frame->width * frame->height;
  for (int c = 0; c < num_channels; ++c) {
    frame->data[c] = src;
    frame->linesize[c] = frame->width;
    frame->buf[c] = create_reference_buffer(src, size);
    src += size;
  }
}
} // namespace

AVFramePtr reference_image_buffer(
    AVPixelFormat fmt,
    void* data,
    size_t width,
    size_t height) {
  auto frame = get_video_frame(fmt, width, height, 0);
  switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      ref_interweaved(frame.get(), data, 3);
      break;
    case AV_PIX_FMT_GRAY8:
      ref_interweaved(frame.get(), data, 1);
      break;
    case AV_PIX_FMT_GRAY16:
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

VideoFramesPtr convert_rgb_array(
    const void* data,
    size_t num_frames,
    size_t height,
    size_t width,
    Rational time_base,
    int pts) {
  auto ret = std::make_unique<VideoFrames>((uintptr_t)data, time_base);
  uint8_t* dst = (uint8_t*)data;
  for (size_t i = 0; i < num_frames; ++i) {
    auto frame =
        detail::get_video_frame(AV_PIX_FMT_RGB24, width, height, pts++);
    detail::ref_interweaved(frame.get(), dst, 3);
    ret->push_back(frame.release());
    dst += height * width * 3;
  }
  return ret;
}

} // namespace spdl::core
