#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
template <size_t depth, ElemClass type, bool is_planar>
CPUBufferPtr convert_frames(const FFmpegAudioFrames* frames) {
  size_t num_frames = frames->get_num_frames();
  const auto& fs = frames->get_frames();
  size_t num_channels =
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 2, 100)
      fs[0]->ch_layout.nb_channels
#else
      fs[0]->channels
#endif
      ;

  if constexpr (is_planar) {
    auto buf = cpu_buffer({num_channels, num_frames}, type, depth);
    uint8_t* dst = static_cast<uint8_t*>(buf->data());
    for (int i = 0; i < num_channels; ++i) {
      for (auto frame : fs) {
        int plane_size = depth * frame->nb_samples;
        memcpy(dst, frame->extended_data[i], plane_size);
        dst += plane_size;
      }
    }
    return buf;
  } else {
    auto buf = cpu_buffer({num_frames, num_channels}, type, depth);
    uint8_t* dst = static_cast<uint8_t*>(buf->data());
    for (auto frame : fs) {
      int plane_size = depth * frame->nb_samples * num_channels;
      memcpy(dst, frame->extended_data[0], plane_size);
      dst += plane_size;
    }
    return buf;
  }
}

CPUBufferPtr convert_audio_frames(const FFmpegAudioFrames* frames) {
  const auto& fs = frames->get_frames();
  if (!fs.size()) {
    SPDL_FAIL("No audio frame to convert to buffer.");
  }
  auto sample_fmt = static_cast<AVSampleFormat>(fs[0]->format);
  switch (sample_fmt) {
    case AV_SAMPLE_FMT_U8:
      return convert_frames<1, ElemClass::UInt, false>(frames);
    case AV_SAMPLE_FMT_U8P:
      return convert_frames<1, ElemClass::UInt, true>(frames);
    case AV_SAMPLE_FMT_S16:
      return convert_frames<2, ElemClass::Int, false>(frames);
    case AV_SAMPLE_FMT_S16P:
      return convert_frames<2, ElemClass::Int, true>(frames);
    case AV_SAMPLE_FMT_S32:
      return convert_frames<4, ElemClass::Int, false>(frames);
    case AV_SAMPLE_FMT_S32P:
      return convert_frames<4, ElemClass::Int, true>(frames);
    case AV_SAMPLE_FMT_FLT:
      return convert_frames<4, ElemClass::Float, false>(frames);
    case AV_SAMPLE_FMT_FLTP:
      return convert_frames<4, ElemClass::Float, true>(frames);
    case AV_SAMPLE_FMT_S64:
      return convert_frames<8, ElemClass::Int, false>(frames);
    case AV_SAMPLE_FMT_S64P:
      return convert_frames<8, ElemClass::Int, true>(frames);
    case AV_SAMPLE_FMT_DBL:
      return convert_frames<8, ElemClass::Float, false>(frames);
    case AV_SAMPLE_FMT_DBLP:
      return convert_frames<8, ElemClass::Float, true>(frames);
    default:
      SPDL_FAIL_INTERNAL(fmt::format(
          "Unexpected sample format: {}", av_get_sample_fmt_name(sample_fmt)));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////
namespace {
void copy_2d(
    uint8_t* src,
    int height,
    int width,
    int src_linesize,
    uint8_t** dst,
    int dst_linesize) {
  TRACE_EVENT("decoding", "conversion::copy_2d");
  for (int h = 0; h < height; ++h) {
    memcpy(*dst, src, width);
    src += src_linesize;
    *dst += dst_linesize;
  }
}

void copy_interleaved(
    const std::vector<AVFrame*>& frames,
    uint8_t* dst,
    unsigned int num_channels,
    size_t w,
    size_t h) {
  size_t wc = num_channels * w;
  for (const auto& f : frames) {
    copy_2d(f->data[0], f->height, wc, f->linesize[0], &dst, wc);
  }
}

template <MediaType media_type>
CPUBufferPtr convert_interleaved(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    size_t num_channels,
    size_t num_frames,
    size_t w,
    size_t h) {
  auto buf = cpu_buffer({batch.size(), num_frames, h, w, num_channels});
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
    size_t h) {
  for (const auto& f : frames) {
    for (size_t c = 0; c < num_planes; ++c) {
      copy_2d(f->data[c], h, w, f->linesize[c], &dst, w);
    }
  }
}

template <MediaType media_type>
CPUBufferPtr convert_planer(
    const std::vector<const FFmpegFrames<media_type>*>& batch,
    size_t num_planes,
    size_t num_frames,
    size_t w,
    size_t h) {
  auto buf = cpu_buffer({batch.size(), num_frames, num_planes, h, w});
  auto dst = (uint8_t*)buf->data();
  for (auto& frames : batch) {
    copy_planer(frames->get_frames(), dst, num_planes, w, h);
    dst += num_frames * num_planes * h * w;
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
    size_t num_frames,
    size_t w,
    size_t h) {
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer({batch.size(), num_frames, 1, h + h2, w});
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
    size_t num_frames,
    size_t w,
    size_t h) {
  assert(w % 2 == 0);

  auto buf = cpu_buffer({batch.size(), num_frames, 1, h + h, w});

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
    size_t num_frames,
    size_t w,
    size_t h) {
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto buf = cpu_buffer({batch.size(), num_frames, 1, h + h2, w});
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
    const std::vector<const FFmpegFrames<media_type>*>& batch) {
  TRACE_EVENT("decoding", "core::convert_frames");
  check_batch_frame_consistency(batch);

  auto ret = [&]() {
    auto& ref_frames = batch.at(0)->get_frames();
    auto w = ref_frames.at(0)->width, h = ref_frames.at(0)->height;
    auto num_frames = ref_frames.size();
    auto pix_fmt = static_cast<AVPixelFormat>(ref_frames.at(0)->format);
    switch (pix_fmt) {
      case AV_PIX_FMT_GRAY8:
        // Technically, not a planer format, but it's the same.
        return convert_planer(batch, 1, num_frames, w, h);
      case AV_PIX_FMT_RGBA:
        return convert_interleaved(batch, 4, num_frames, w, h);
      case AV_PIX_FMT_RGB24:
        return convert_interleaved(batch, 3, num_frames, w, h);
      case AV_PIX_FMT_YUVJ444P:
      case AV_PIX_FMT_YUV444P:
        return convert_planer(batch, 3, num_frames, w, h);
      case AV_PIX_FMT_YUVJ420P:
      case AV_PIX_FMT_YUV420P:
        return convert_yuv420p(batch, num_frames, w, h);
      case AV_PIX_FMT_YUVJ422P:
      case AV_PIX_FMT_YUV422P:
        return convert_yuv422p(batch, num_frames, w, h);
      case AV_PIX_FMT_NV12: {
        return convert_nv12(batch, num_frames, w, h);
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
    const std::vector<const FFmpegImageFrames*>& batch);

template CPUBufferPtr convert_frames(
    const std::vector<const FFmpegVideoFrames*>& batch);

} // namespace spdl::core
