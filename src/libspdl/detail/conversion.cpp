#include <libspdl/detail/conversion.h>

#include <fmt/core.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
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

VideoBuffer convert_interleaved(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;

  VideoBuffer buf{frames.size(), 3, h, w, /*channel_last=*/true};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  size_t wc = buf.c * buf.w;
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    copy_2d(f->data[0], f->height, wc, f->linesize[0], &dst, wc);
  }
  return buf;
}

VideoBuffer convert_planer(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;

  VideoBuffer buf{frames.size(), 3, h, w};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    for (int c = 0; c < buf.c; ++c) {
      copy_2d(f->data[c], h, w, f->linesize[c], &dst, w);
    }
  }
  return buf;
}

VideoBuffer convert_plane(const std::vector<AVFrame*>& frames, int plane) {
  size_t h = frames[0]->height, w = frames[0]->width;

  VideoBuffer buf{frames.size(), 1, h, w};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    copy_2d(f->data[plane], h, w, f->linesize[plane], &dst, w);
  }
  return buf;
}

VideoBuffer convert_yuv420p(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  VideoBuffer buf{frames.size(), 1, h + h2, w};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
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

VideoBuffer convert_u_or_v(const std::vector<AVFrame*>& frames, int plane) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  VideoBuffer buf{frames.size(), 1, h2, w2};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    copy_2d(f->data[plane], h2, w2, f->linesize[plane], &dst, w2);
  }
  return buf;
}

VideoBuffer convert_nv12(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2;

  VideoBuffer buf{frames.size(), 1, h + h2, w};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    // Y
    copy_2d(f->data[0], h, w, f->linesize[0], &dst, w);
    // UV
    copy_2d(f->data[1], h2, w, f->linesize[1], &dst, w);
  }
  return buf;
}

VideoBuffer convert_nv12_uv(const std::vector<AVFrame*>& frames) {
  size_t h = frames[0]->height, w = frames[0]->width;
  assert(h % 2 == 0 && w % 2 == 0);
  size_t h2 = h / 2, w2 = w / 2;

  VideoBuffer buf{frames.size(), 2, h2, w2, /*channel_last=*/true};
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  uint8_t* dst = buf.data.data();
  for (const auto& f : frames) {
    copy_2d(f->data[1], h2, w, f->linesize[1], &dst, w);
  }
  return buf;
}

VideoBuffer convert_video_frames_cuda(
    const std::vector<AVFrame*>& frames,
    const int plane) {
  auto frames_ctx = (AVHWFramesContext*)(frames[0]->hw_frames_ctx->data);
  auto sw_pix_fmt = frames_ctx->sw_format;
  throw std::runtime_error(fmt::format(
      "CUDA frame ({}) is not supported.", av_get_pix_fmt_name(sw_pix_fmt)));
}

VideoBuffer convert_video_frames_cpu(
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
          throw std::runtime_error(fmt::format(
              "`plane` value for RGB24 format must be 0. (Found: {})", plane));
      }
    }
    case AV_PIX_FMT_YUV444P: {
      switch (plane) {
        case -1:
          return convert_interleaved(frames);
        case 0:
        case 1:
        case 2:
          return convert_plane(frames, plane);
        default:
          throw std::runtime_error(fmt::format(
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
          throw std::runtime_error(fmt::format(
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
          throw std::runtime_error(fmt::format(
              "Valid `plane` values for NV12 are [0, 1]. (Found: {})", plane));
      }
    }
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported pixel format: {}", av_get_pix_fmt_name(pix_fmt)));
  }
}

} // namespace

VideoBuffer convert_video_frames(
    const std::vector<AVFrame*>& frames,
    const int plane) {
  if (!frames.size()) {
    return {};
  }
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  if (pix_fmt == AV_PIX_FMT_CUDA) {
    return convert_video_frames_cuda(frames, plane);
  }
  return convert_video_frames_cpu(frames, plane);
}

} // namespace spdl::detail
