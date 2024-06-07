#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/ffmpeg/encoding.h"
#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace spdl::core {
namespace {
std::tuple<size_t, size_t> get_image_size(
    const AVPixelFormat pix_fmt,
    const std::vector<size_t>& shape,
    int bit_depth) {
  switch (pix_fmt) {
    case AV_PIX_FMT_GRAY8: {
      if (shape.size() != 2) {
        SPDL_FAIL(fmt::format(
            "Array must be 2D when pixel format is \"gray\", but found {}D",
            shape.size()));
      }
      if (bit_depth != 8) {
        SPDL_FAIL(fmt::format(
            "Pixel must be 8 bit when pixel format is \"gray\", but found {}",
            bit_depth));
      }
      return {shape[1], shape[0]};
    }
    case AV_PIX_FMT_GRAY16BE: {
      if (shape.size() != 2) {
        SPDL_FAIL(fmt::format(
            "Array must be 2D when pixel format is \"gray16be\", but found {}D",
            shape.size()));
      }
      if (bit_depth != 16) {
        SPDL_FAIL(fmt::format(
            "Pixel must be 16 bit when pixel format is \"gray16be\", but found {}",
            bit_depth));
      }
      return {shape[1], shape[0]};
    }
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      if (shape.size() != 3) {
        SPDL_FAIL(fmt::format(
            "Array must be 3D when pixel format is \"rgb24\" or \"bgr24\", but found {}D",
            shape.size()));
      }
      if (shape[2] != 3) {
        SPDL_FAIL(fmt::format(
            "Shape must be [height, width, channel==3], but channel=={} found.",
            shape[2]));
      }
      if (bit_depth != 8) {
        SPDL_FAIL(fmt::format(
            "Pixel must be 8 bit when pixel format is \"rgb24\" or \"bgr24\", but found {}",
            bit_depth));
      }
      return {shape[1], shape[0]};
    }
    case AV_PIX_FMT_YUV444P:
      if (shape.size() != 3) {
        SPDL_FAIL(fmt::format(
            "Array must be 3D when pixel format is \"yuv444p\", but found {}D",
            shape.size()));
      }
      if (shape[0] != 3) {
        SPDL_FAIL(fmt::format(
            "Shape must be [channel==3, height, width], but channel=={} found.",
            shape[0]));
      }
      if (bit_depth != 8) {
        SPDL_FAIL(fmt::format(
            "Array must be 8 bit when pixel format is \"YUV444P\", but found {}",
            bit_depth));
      }
      return {shape[2], shape[1]};
    default:
      SPDL_FAIL(fmt::format(
          "Unsupported source pixel format: {}", av_get_pix_fmt_name(pix_fmt)));
  }
}
} // namespace

void encode_image(
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    int bit_depth,
    const std::string& src_pix_fmt,
    const std::optional<EncodeConfig>& enc_cfg) {
  const AVPixelFormat src_fmt = av_get_pix_fmt(src_pix_fmt.c_str());
  if (src_fmt == AV_PIX_FMT_NONE) {
    SPDL_FAIL(fmt::format("Invalid source pixel format: {}", src_pix_fmt));
  }
  if (src_fmt == AV_PIX_FMT_GRAY16BE) {
    if (!enc_cfg || !(enc_cfg->format)) {
      LOG_FIRST_N(WARNING, 1)
          << "The source format is gray16be, but the output format is not specified. "
             "The default pixel format of the encoder will be used. If you intend to "
             "save data with gray16be, then specify `EncodeConfig.format=\"gray16be\"`.";
    }
  }
  auto [src_width, src_height] = get_image_size(src_fmt, shape, bit_depth);
  auto [encoder, filter_graph] = detail::get_encode_process(
      uri, src_fmt, src_width, src_height, enc_cfg.value_or(EncodeConfig{}));

  auto frame =
      detail::reference_image_buffer(src_fmt, data, src_width, src_height);
  auto filtering = filter_graph.filter(frame.get());
  while (filtering) {
    encoder.encode(filtering());
  }
  encoder.encode(nullptr);
}

} // namespace spdl::core
