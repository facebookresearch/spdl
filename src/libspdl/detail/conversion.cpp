#include <libspdl/detail/conversion.h>
#include <libspdl/logging.h>
#include <libspdl/types.h>
#ifdef SPDL_USE_CUDA
#include <libspdl/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>

#ifdef SPDL_USE_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
}

namespace spdl::detail {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////
void copy_2d(
    uint8_t* src,
    int height,
    int width,
    int src_linesize,
    uint8_t** dst,
    int dst_linesize) {
  for (int h = 0; h < height; ++h) {
    memcpy(*dst, src, width);
    src += src_linesize;
    *dst += dst_linesize;
  }
}

Buffer convert_interleaved(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;

  Buffer buf = cpu_buffer({frames.size(), h, w, 3}, true);
  size_t wc = 3 * w;
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    copy_2d(f->data[0], f->height, wc, f->linesize[0], &dst, wc);
  }
  return buf;
}

Buffer convert_planer(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;

  Buffer buf = cpu_buffer({frames.size(), 3, h, w});
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    for (int c = 0; c < 3; ++c) {
      copy_2d(f->data[c], h, w, f->linesize[c], &dst, w);
    }
  }
  return buf;
}

Buffer convert_plane(const std::vector<AVFrame*>& frames, int plane) {
  size_t h = frames[0]->height, w = frames[0]->width;

  Buffer buf = cpu_buffer({frames.size(), 1, h, w});
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    copy_2d(f->data[plane], h, w, f->linesize[plane], &dst, w);
  }
  return buf;
}

Buffer convert_yuv420p(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  Buffer buf = cpu_buffer({frames.size(), 1, h + h2, w});
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    uint8_t* dst2 = dst + w2;
    copy_2d(f->data[1], h2, w2, f->linesize[1], &dst, w);
    copy_2d(f->data[2], h2, w2, f->linesize[2], &dst2, w);
  }
  return buf;
}

Buffer convert_u_or_v(const std::vector<AVFrame*>& frames, int plane) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  Buffer buf = cpu_buffer({frames.size(), 1, h2, w2});
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    copy_2d(f->data[plane], h2, w2, f->linesize[plane], &dst, w2);
  }
  return buf;
}

Buffer convert_nv12(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  Buffer buf = cpu_buffer({frames.size(), 1, h + h2, w});
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    copy_2d(f->data[1], h2, w, f->linesize[1], &dst, w);
  }
  return buf;
}

Buffer convert_nv12_uv(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  Buffer buf = cpu_buffer({frames.size(), h2, w2, 2}, true);
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    copy_2d(f->data[1], h2, w, f->linesize[1], &dst, w);
  }
  return buf;
}

#ifdef SPDL_USE_CUDA
Buffer convert_nv12_cuda(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  auto hw_frames_ctx = (AVHWFramesContext*)frames[0]->hw_frames_ctx->data;
  auto hw_device_ctx = (AVHWDeviceContext*)hw_frames_ctx->device_ctx;
  auto cuda_device_ctx = (AVCUDADeviceContext*)hw_device_ctx->hwctx;
  auto stream = cuda_device_ctx->stream;
  XLOG(DBG9) << "CUcontext: " << cuda_device_ctx->cuda_ctx;
  XLOG(DBG9) << "CUstream: " << cuda_device_ctx->stream;

  XLOG(DBG) << "creating cuda buffer";
  Buffer buf = cuda_buffer({frames.size(), 1, h + h2, w}, stream);
  uint8_t* dst = static_cast<uint8_t*>(buf.data());
  for (const auto& f : frames) {
    // Y
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst,
        w,
        f->data[0],
        f->linesize[0],
        w,
        h,
        cudaMemcpyDeviceToDevice,
        stream));
    dst += h * w;
    // UV
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst,
        w,
        f->data[1],
        f->linesize[1],
        w,
        h2,
        cudaMemcpyDeviceToDevice,
        stream));
    dst += h2 * w;
  }
  return buf;
}

Buffer convert_video_frames_cuda(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index) {
  auto frames_ctx = (AVHWFramesContext*)(frames[0]->hw_frames_ctx->data);
  auto sw_pix_fmt = frames_ctx->sw_format;
  switch (sw_pix_fmt) {
    case AV_PIX_FMT_NV12:
      if (index) {
        SPDL_FAIL(fmt::format(
            "Selecting a plane from CUDA frame ({}) is not supported.",
            av_get_pix_fmt_name(sw_pix_fmt)));
      }
      return convert_nv12_cuda(frames);
    default:
      SPDL_FAIL(fmt::format(
          "CUDA frame ({}) is not supported.",
          av_get_pix_fmt_name(sw_pix_fmt)));
  }
}
#endif

Buffer convert_video_frames_cpu(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index) {
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  switch (pix_fmt) {
    case AV_PIX_FMT_RGB24: {
      if (auto plane = index.value_or(0); plane != 0) {
        SPDL_FAIL(fmt::format(
            "Valid `plane` value for RGB24 is 0. (Found: {})", plane));
      }
      return convert_interleaved(frames);
    }
    case AV_PIX_FMT_YUV444P: {
      if (!index) {
        return convert_planer(frames);
      }
      auto plane = index.value();
      switch (plane) {
        case 0:
        case 1:
        case 2:
          return convert_plane(frames, plane);
        default:
          SPDL_FAIL(fmt::format(
              "Valid `plane` values for YUV444P are [0, 1, 2]. (Found: {})",
              plane));
      }
    }
    case AV_PIX_FMT_YUV420P: {
      if (!index) {
        return convert_yuv420p(frames);
      }
      auto plane = index.value();
      switch (plane) {
        case 0:
          return convert_plane(frames, plane);
        case 1:
        case 2:
          return convert_u_or_v(frames, plane);
        default:
          SPDL_FAIL(fmt::format(
              "Valid `plane` values for YUV420P are [0, 1, 2]. (Found: {})",
              plane));
      }
    }
    case AV_PIX_FMT_NV12: {
      if (!index) {
        return convert_nv12(frames);
      }
      auto plane = index.value();
      switch (plane) {
        case 0:
          return convert_plane(frames, plane);
        case 1:
          return convert_nv12_uv(frames);
        default:
          SPDL_FAIL(fmt::format(
              "Valid `plane` values for NV12 are [0, 1]. (Found: {})", plane));
      }
    }
    default:
      SPDL_FAIL(fmt::format(
          "Unsupported pixel format: {}", av_get_pix_fmt_name(pix_fmt)));
  }
}

Buffer convert_video_frames(
    const Frames& frames,
    const std::optional<int>& index) {
  if (frames.type != MediaType::Video) {
    SPDL_FAIL("Frames must be video type.");
  }

  const auto& fs = frames.frames;
  if (!fs.size()) {
    SPDL_FAIL("No video frame to convert to buffer.");
  }
  auto pix_fmt = static_cast<AVPixelFormat>(fs[0]->format);
  if (pix_fmt == AV_PIX_FMT_CUDA) {
#ifdef SPDL_USE_CUDA
    return convert_video_frames_cuda(fs, index);
#else
    SPDL_FAIL("SPDL is not compiled with CUDA support.");
#endif
  }
  return convert_video_frames_cpu(fs, index);
}

template <size_t depth, ElemClass type, bool is_planar>
Buffer convert_audio_frames(
    const Frames& frames,
    const std::optional<int>& index) {
  size_t num_frames = frames.get_num_samples();
  size_t num_channels = frames.frames[0]->channels;

  if (index) {
    auto c = index.value();
    if (c < 0 || c >= num_channels) {
      SPDL_FAIL(fmt::format("Channel index must be [0, {})", num_channels));
    }
    num_channels = 1;
  }

  if constexpr (is_planar) {
    auto buffer =
        cpu_buffer({num_channels, num_frames}, !is_planar, type, depth);
    uint8_t* dst = static_cast<uint8_t*>(buffer.data());
    for (int i = 0; i < num_channels; ++i) {
      if (index && index.value() != i) {
        continue;
      }
      for (auto frame : frames.frames) {
        int plane_size = depth * frame->nb_samples;
        memcpy(dst, frame->extended_data[i], plane_size);
        dst += plane_size;
      }
    }
    return buffer;
  } else {
    if (index) {
      SPDL_FAIL("Cannot select channel from non-planar audio.");
    }
    auto buffer =
        cpu_buffer({num_frames, num_channels}, !is_planar, type, depth);
    uint8_t* dst = static_cast<uint8_t*>(buffer.data());
    for (auto frame : frames.frames) {
      int plane_size = depth * frame->nb_samples * num_channels;
      memcpy(dst, frame->extended_data[0], plane_size);
      dst += plane_size;
    }
    return buffer;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
Buffer convert_audio_frames(const Frames& frames, const std::optional<int>& i) {
  if (frames.type != MediaType::Audio) {
    SPDL_FAIL("Frames must be audio type.");
  }
  const auto& fs = frames.frames;
  if (!fs.size()) {
    SPDL_FAIL("No audio frame to convert to buffer.");
  }

  // NOTE:
  // This conversion converts all the samples in underlying frames.
  // This does not take the time stamp of each sample into account.
  //
  // TODO:
  // Check time stamp here?
  auto sample_fmt = static_cast<AVSampleFormat>(fs[0]->format);
  switch (sample_fmt) {
    case AV_SAMPLE_FMT_U8:
      return convert_audio_frames<1, ElemClass::UInt, false>(frames, i);
    case AV_SAMPLE_FMT_U8P:
      return convert_audio_frames<1, ElemClass::UInt, true>(frames, i);
    case AV_SAMPLE_FMT_S16:
      return convert_audio_frames<2, ElemClass::Int, false>(frames, i);
    case AV_SAMPLE_FMT_S16P:
      return convert_audio_frames<2, ElemClass::Int, true>(frames, i);
    case AV_SAMPLE_FMT_S32:
      return convert_audio_frames<4, ElemClass::Int, false>(frames, i);
    case AV_SAMPLE_FMT_S32P:
      return convert_audio_frames<4, ElemClass::Int, true>(frames, i);
    case AV_SAMPLE_FMT_FLT:
      return convert_audio_frames<4, ElemClass::Float, false>(frames, i);
    case AV_SAMPLE_FMT_FLTP:
      return convert_audio_frames<4, ElemClass::Float, true>(frames, i);
    case AV_SAMPLE_FMT_S64:
      return convert_audio_frames<8, ElemClass::Int, false>(frames, i);
    case AV_SAMPLE_FMT_S64P:
      return convert_audio_frames<8, ElemClass::Int, true>(frames, i);
    case AV_SAMPLE_FMT_DBL:
      return convert_audio_frames<8, ElemClass::Float, false>(frames, i);
    case AV_SAMPLE_FMT_DBLP:
      return convert_audio_frames<8, ElemClass::Float, true>(frames, i);
    default:
      SPDL_FAIL_INTERNAL(fmt::format(
          "Unexpected sample format: {}", av_get_sample_fmt_name(sample_fmt)));
  }
}
} // namespace

Buffer convert_frames(const Frames& frames, const std::optional<int>& index) {
  switch (frames.type) {
    case MediaType::Audio:
      return detail::convert_audio_frames(frames, index);
    case MediaType::Video:
      return detail::convert_video_frames(frames, index);
  }
}

} // namespace spdl::detail
