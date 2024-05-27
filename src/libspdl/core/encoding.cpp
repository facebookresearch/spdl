#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/ffmpeg/encoding.h"
#include "libspdl/core/detail/logging.h"

#include <fmt/core.h>

#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace spdl::core {
namespace {
std::tuple<size_t, size_t> get_image_size(
    const AVPixelFormat pix_fmt,
    const std::vector<size_t>& shape) {
  switch (pix_fmt) {
    case AV_PIX_FMT_GRAY8: {
      if (shape.size() != 2) {
        SPDL_FAIL(fmt::format(
            "Array must be 2D when pixel format is \"gray\", but found {}D",
            shape.size()));
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
      return {shape[1], shape[0]};
    }
    case AV_PIX_FMT_YUV444P:
      if (shape.size() != 3) {
        SPDL_FAIL(fmt::format(
            "Array must be 3D when pixel format is \"YUV444P\", but found {}D",
            shape.size()));
      }
      if (shape[0] != 3) {
        SPDL_FAIL(fmt::format(
            "Shape must be [channel==3, height, width], but channel=={} found.",
            shape[0]));
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
    const std::string& src_pix_fmt,
    const std::optional<EncodeConfig>& enc_cfg) {
  const AVPixelFormat src_fmt = av_get_pix_fmt(src_pix_fmt.c_str());
  if (src_fmt == AV_PIX_FMT_NONE) {
    SPDL_FAIL(fmt::format("Invalid source pixel format: {}", src_pix_fmt));
  }
  auto [src_width, src_height] = get_image_size(src_fmt, shape);
  auto [encoder, filter_graph] = detail::get_encode_process(
      uri, src_fmt, src_width, src_height, enc_cfg.value_or(EncodeConfig{}));

  auto frame =
      detail::reference_image_buffer(src_fmt, data, src_width, src_height);
  for (auto filtered : filter_graph.filter(frame.get())) {
    encoder.encode(filtered);
  }
  encoder.encode(nullptr);
}

} // namespace spdl::core
