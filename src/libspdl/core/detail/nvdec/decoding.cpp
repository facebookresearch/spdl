/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/nvdec/decoding.h"

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/nvdec/buffer.h"
#include "libspdl/core/detail/nvdec/decoder.h"
#include "libspdl/core/detail/nvdec/utils.h"
#include "libspdl/core/detail/nvdec/wrapper.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <cuviddec.h>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace spdl::core::detail {
namespace {
struct _Decoder {
  NvDecDecoderInternal decoder{};
  bool decoding_ongoing = false;
};

_Decoder& get_decoder() {
  // Decoder objects are thread local so that they survive each decoding job,
  // which gives us opportunity to reuse the decoder and avoid destruction and
  // recreation ops, which are very expensive (~300ms).
  static thread_local _Decoder decoder{};
  return decoder;
}

} // namespace

CUDABufferPtr get_buffer(
    const CUDAConfig& cuda_config,
    size_t num_packets,
    AVCodecParameters* codecpar,
    const CropArea& crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt) {
  size_t w = target_width > 0 ? target_width
                              : (codecpar->width - crop.left - crop.right);
  size_t h = target_height > 0 ? target_height
                               : (codecpar->height - crop.top - crop.bottom);

  size_t c;
  auto pix_fmt_val = pix_fmt.value_or("nv12");
  if (pix_fmt_val == "nv12") {
    c = 1;
    h = h + h / 2;
  } else if (pix_fmt_val == "rgba" || pix_fmt_val == "bgra") {
    c = 4;
  } else {
    SPDL_FAIL(fmt::format("Unsupported pixel format: {}", pix_fmt_val));
  }
  return cuda_buffer(std::vector<size_t>{num_packets, c, h, w}, cuda_config);
}

template <MediaType media_type>
CUDABufferPtr decode_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt) {
  size_t num_packets = packets->num_packets();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  ensure_cuda_initialized();

  TRACE_EVENT("nvdec", "decode_packets");

  _Decoder& _dec = get_decoder();

  if (_dec.decoding_ongoing) {
    // When the previous decoding ended with an error, if the new input data is
    // the same format, then handle_video_sequence might not be called because
    // NVDEC decoder thinks that incoming frames are from the same sequence.
    // This can cause strange decoder behavior. So we need to reset the decoder.
    _dec.decoder.reset();
  }
  _dec.decoder.init(
      cuda_config.device_index,
      packets->codecpar->codec_id,
      packets->time_base,
      packets->timestamp,
      crop,
      target_width,
      target_height,
      pix_fmt);

  _dec.decoding_ongoing = true;

  auto ret = _dec.decoder.decode(
      std::move(packets),
      cuda_config,
      crop,
      target_width,
      target_height,
      pix_fmt);

  _dec.decoding_ongoing = false;

  return std::move(ret);
}

template CUDABufferPtr decode_nvdec(
    VideoPacketsPtr packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

} // namespace spdl::core::detail
