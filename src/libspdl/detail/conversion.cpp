#include <libspdl/detail/conversion.h>
#include <libspdl/logging.h>
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
    const int plane) {
  auto frames_ctx = (AVHWFramesContext*)(frames[0]->hw_frames_ctx->data);
  auto sw_pix_fmt = frames_ctx->sw_format;
  switch (sw_pix_fmt) {
    case AV_PIX_FMT_NV12:
      switch (plane) {
        case -1:
          return convert_nv12_cuda(frames);
        default:
          SPDL_FAIL(fmt::format(
              "CUDA frame ({}:{}) is not supported.",
              av_get_pix_fmt_name(sw_pix_fmt),
              plane));
      }
    default:
      SPDL_FAIL(fmt::format(
          "CUDA frame ({}) is not supported.",
          av_get_pix_fmt_name(sw_pix_fmt)));
  }
}
#endif

Buffer convert_video_frames_cpu(
    const std::vector<AVFrame*>& frames,
    const int plane) {
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  switch (pix_fmt) {
    case AV_PIX_FMT_RGB24: {
      switch (plane) {
        case -1:
        case 0:
          return convert_interleaved(frames);
        default:
          SPDL_FAIL(fmt::format(
              "`plane` value for RGB24 format must be 0. (Found: {})", plane));
      }
    }
    case AV_PIX_FMT_YUV444P: {
      switch (plane) {
        case -1:
          return convert_planer(frames);
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
      switch (plane) {
        case -1:
          return convert_yuv420p(frames);
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
      switch (plane) {
        case -1:
          return convert_nv12(frames);
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

} // namespace

Buffer convert_video_frames(const Frames& frames, const int plane) {
  if (frames.type != Frames::Type::Video) {
    SPDL_FAIL("Frames class is not video type.");
  }

  const auto& fs = frames.frames;
  if (!fs.size()) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto pix_fmt = static_cast<AVPixelFormat>(fs[0]->format);
  if (pix_fmt == AV_PIX_FMT_CUDA) {
#ifdef SPDL_USE_CUDA
    return convert_video_frames_cuda(fs, plane);
#else
    SPDL_FAIL("SPDL is not compiled with CUDA support.");
#endif
  }
  return convert_video_frames_cpu(fs, plane);
}

} // namespace spdl::detail
