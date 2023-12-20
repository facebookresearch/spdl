#include <libspdl/conversion.h>

#include <fmt/core.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace spdl {

VideoBuffer convert_rgb24(const DecodedFrames& val) {
  assert(val.frames[0]->format == AV_PIX_FMT_RGB24);
  VideoBuffer buf;
  buf.n = val.frames.size();
  buf.c = 3;
  buf.h = static_cast<size_t>(val.frames[0]->height);
  buf.w = static_cast<size_t>(val.frames[0]->width);
  buf.channel_last = true;
  buf.data.resize(buf.n * buf.c * buf.h * buf.w);
  size_t linesize = buf.c * buf.w;
  uint8_t* dst = buf.data.data();
  for (const auto& frame : val.frames) {
    uint8_t* src = frame->data[0];
    for (int i = 0; i < frame->height; ++i) {
      memcpy(dst, src, linesize);
      src += frame->linesize[0];
      dst += linesize;
    }
  }
  return buf;
}

VideoBuffer convert_yuv420p(const DecodedFrames& val) {
  assert(val.frames[0]->format == AV_PIX_FMT_YUV420P);
  size_t height = val.frames[0]->height;
  size_t width = val.frames[0]->width;
  assert(height % 2 == 0 && width % 2 == 0);
  size_t h2 = height / 2;
  size_t w2 = width / 2;
  VideoBuffer buf;
  buf.n = val.frames.size();
  buf.c = 1;
  buf.h = height + h2;
  buf.w = width;
  buf.channel_last = false;
  buf.data.resize(buf.n * buf.h * buf.w);

  uint8_t* dst = buf.data.data();
  for (const auto& frame : val.frames) {
    // Y
    {
      uint8_t* src = frame->data[0];
      size_t linesize = buf.w;
      for (int i = 0; i < frame->height; ++i) {
        memcpy(dst, src, linesize);
        src += frame->linesize[0];
        dst += linesize;
      }
    }
    // U & V
    {
      uint8_t* src_u = frame->data[1];
      uint8_t* src_v = frame->data[2];
      size_t linesize = w2;
      for (int i = 0; i < h2; ++i) {
        // U
        memcpy(dst, src_u, linesize);
        src_u += frame->linesize[1];
        dst += linesize;
        // V
        memcpy(dst, src_v, linesize);
        src_v += frame->linesize[2];
        dst += linesize;
      }
    }
  }
  return buf;
}

VideoBuffer convert_nv12(const DecodedFrames& val) {
  assert(val.frames[0]->format == AV_PIX_FMT_NV12);
  size_t height = val.frames[0]->height;
  size_t width = val.frames[0]->width;
  assert(height % 2 == 0 && width % 2 == 0);
  size_t h2 = height / 2;
  size_t w2 = width / 2;
  VideoBuffer buf;
  buf.n = val.frames.size();
  buf.c = 1;
  buf.h = height + h2;
  buf.w = width;
  buf.channel_last = false;
  buf.data.resize(buf.n * buf.h * buf.w);

  uint8_t* dst = buf.data.data();
  for (const auto& frame : val.frames) {
    // Y
    {
      uint8_t* src = frame->data[0];
      size_t linesize = buf.w;
      for (int i = 0; i < frame->height; ++i) {
        memcpy(dst, src, linesize);
        src += frame->linesize[0];
        dst += linesize;
      }
    }
    // UV
    // TODO: Fix the interweived UV
    {
      uint8_t* src = frame->data[1];
      size_t linesize = buf.w;
      for (int i = 0; i < h2; ++i) {
        memcpy(dst, src, linesize);
        src += frame->linesize[1];
        dst += linesize;
      }
    }
  }
  return buf;
}

VideoBuffer convert_frames(const DecodedFrames& val) {
  switch (static_cast<AVPixelFormat>(val.frames[0]->format)) {
    case AV_PIX_FMT_RGB24:
      return convert_rgb24(val);
    case AV_PIX_FMT_YUV420P:
      return convert_yuv420p(val);
    case AV_PIX_FMT_NV12:
      return convert_nv12(val);
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported pixel format: {}",
          av_get_pix_fmt_name(
              static_cast<AVPixelFormat>(val.frames[0]->format))));
  }
}

} // namespace spdl
