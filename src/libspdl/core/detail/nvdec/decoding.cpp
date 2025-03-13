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

CUDABufferTracker get_buffer_tracker(
    const CUDAConfig& cuda_config,
    size_t num_packets,
    AVCodecParameters* codecpar,
    const CropArea& crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool is_image) {
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

  auto shape = is_image ? std::vector<size_t>{c, h, w}
                        : std::vector<size_t>{num_packets, c, h, w};

  return CUDABufferTracker{cuda_config, shape};
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

template CUDABufferPtr decode_nvdec(
    ImagePacketsPtr packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

struct _DecoderLegacy {
  NvDecDecoderCore decoder{};
  bool decoding_ongoing = false;
};

_DecoderLegacy& get_decoder_legacy() {
  // Decoder objects are thread local so that they survive each decoding job,
  // which gives us opportunity to reuse the decoder and avoid destruction and
  // recreation ops, which are very expensive (~300ms).
  static thread_local _DecoderLegacy decoder{};
  return decoder;
}

CUDABufferPtr decode_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool strict) {
  size_t num_packets = packets.size();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  ensure_cuda_initialized();

  auto& p0 = packets[0];
  if (!pix_fmt) {
    for (auto& packet : packets) {
      if (packet->codecpar->format != p0->codecpar->format) {
        SPDL_FAIL(fmt::format(
            "All images must have the same pixel format. The first one has {}, while another has {}.",
            av_get_pix_fmt_name((AVPixelFormat)p0->codecpar->format),
            av_get_pix_fmt_name((AVPixelFormat)packet->codecpar->format)));
      }
    }
  }
  for (int i = num_packets - 1; i >= 0; --i) {
    // An example of YUVJ411P format is "ILSVRC2012_val_00016550.JPEG" from
    // ImageNet validation set.
    // When such a data is fed to NVDEC, it just does not call the callback.
    if ((AVPixelFormat)packets[i]->codecpar->format == AV_PIX_FMT_YUVJ411P) {
      auto msg =
          fmt::format("Found unsupported format YUVJ411P at index {}.", i);
      if (strict) {
        SPDL_FAIL(msg);
      } else {
        LOG(ERROR) << msg;
        packets.erase(packets.begin() + i);
      }
    }
  }

  TRACE_EVENT("nvdec", "decode_packets");

  _DecoderLegacy& _dec = get_decoder_legacy();

  auto tracker = get_buffer_tracker(
      cuda_config,
      num_packets,
      p0->codecpar,
      crop,
      target_width,
      target_height,
      pix_fmt,
      false);
  _dec.decoder.tracker = &tracker;

  auto decode_fn = [&](ImagePacketsPtr& packet) {
    if (_dec.decoding_ongoing) {
      // When the previous decoding ended with an error, if the new input data
      // is the same format, then handle_video_sequence might not be called
      // because NVDEC decoder thinks that incoming frames are from the same
      // sequence. This can cause strange decoder behavior. So we need to reset
      // the decoder.
      _dec.decoder.reset();
    }
    _dec.decoder.init(
        cuda_config.device_index,
        covert_codec_id(packet->codecpar->codec_id),
        packet->time_base,
        packet->timestamp,
        crop,
        target_width,
        target_height,
        pix_fmt);

    _dec.decoding_ongoing = true;
    unsigned long flags =
        CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE | CUVID_PKT_ENDOFSTREAM;

    auto pkt = packet->get_packets()[0];
    _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);
    _dec.decoding_ongoing = false;
  };

  for (auto& packet : packets) {
    int i_ = tracker.i;
    try {
      decode_fn(packet);
    } catch (std::exception& e) {
      LOG(ERROR) << fmt::format("Failed to decode image: {}", e.what());
      if (strict) {
        throw;
      }
    }
    // When a media is not supported, NVDEC does not necessarily
    // fail. Instead it just does not call the callback.
    // So we need to check if the number of decoded frames has changed.
    if (i_ == tracker.i) {
      auto msg =
          "Failed to decode image. (This could be due to unsupported media type. "
          "NVDEC does not always fail even when the input format is not supported.)";
      if (strict) {
        SPDL_FAIL(msg);
      } else {
        LOG(ERROR) << msg;
      }
    }
  }

  return std::move(tracker.buffer);
}

} // namespace spdl::core::detail
