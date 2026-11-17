/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "crop_area.h"
#include "register_spdl_cuda_extensions.h"
#include "spdl/io/lib/fused/external.h"

#include <libspdl/core/bsf.h>
#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/nvdec/decoder.h>
#include <libspdl/cuda/transfer.h>
#include <libspdl/cuda/types.h>

#ifdef SPDL_USE_CUDA
#include <libspdl/cuda/color_conversion.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace spdl::cuda {
namespace {
#ifdef SPDL_USE_CUDA
struct CUDAValue : fused::ExternalValue {
  CUDABufferPtr buffer;

  explicit CUDAValue(CUDABufferPtr value) : buffer{std::move(value)} {}

  nb::object to_python() override {
    return nb::cast(std::move(buffer));
  }
};

template <typename T>
T& input(const std::vector<fused::ExternalInput>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("CUDA fused operations require one input.");
  }
  const auto& value = inputs.front();
  auto* ptr = std::get_if<T*>(&value);
  if (!ptr || !**ptr) {
    throw std::invalid_argument("Unexpected CUDA fused operation input.");
  }
  return **ptr;
}

struct TransferOp : fused::ExternalOp {
  CUDAConfig config;

  explicit TransferOp(const CUDAConfig& cfg) : config{cfg} {}

  fused::ExternalValuePtr run(
      const std::vector<fused::ExternalInput>& inputs) const override {
    return std::make_shared<CUDAValue>(
        transfer_buffer(std::move(input<core::CPUBufferPtr>(inputs)), config));
  }
};

struct ColorOp : fused::ExternalOp {
  CUDAConfig config;
  bool bgr;
  int coeff;
  bool sync;

  ColorOp(const CUDAConfig& cfg, bool bgr_, int coeff_, bool sync_)
      : config{cfg}, bgr{bgr_}, coeff{coeff_}, sync{sync_} {}

  fused::ExternalValuePtr run(
      const std::vector<fused::ExternalInput>& inputs) const override {
    const auto* value =
        dynamic_cast<CUDAValue*>(input<fused::ExternalValuePtr>(inputs).get());
    if (!value || !value->buffer) {
      throw std::invalid_argument(
          "Expected a CUDA buffer for NV12 conversion.");
    }
    return std::make_shared<CUDAValue>(
        bgr ? nv12_to_planar_bgr(*value->buffer, config, coeff, sync)
            : nv12_to_planar_rgb(*value->buffer, config, coeff, sync));
  }
};

#ifdef SPDL_USE_NVCODEC
struct DecodeOp : fused::ExternalOp {
  CUDAConfig config;
  bool bgr;
  CropArea crop;
  int width;
  int height;

  DecodeOp(const CUDAConfig& cfg, bool bgr_, CropArea crop_, int w, int h)
      : config{cfg}, bgr{bgr_}, crop{crop_}, width{w}, height{h} {}

  fused::ExternalValuePtr run(
      const std::vector<fused::ExternalInput>& inputs) const override {
    auto packets = std::move(input<core::VideoPacketsPtr>(inputs));
    if (!packets->codec) {
      throw std::invalid_argument("The packets must have codec.");
    }
    const core::VideoCodec codec{*packets->codec};
    const auto name = codec.get_name();
    if (name == "h264" || name == "hevc") {
      core::BSF<core::MediaType::Video> bsf{codec, name + "_mp4toannexb"};
      auto filtered = bsf.filter(std::move(packets), true);
      if (!filtered) {
        throw std::runtime_error("Bit stream filtering produced no packets.");
      }
      packets = std::move(*filtered);
    }
    // A reentrant Python allocator must not reset the decoder running here.
    // Failed decoders are destroyed before fusion retries the function eagerly.
    thread_local std::unique_ptr<NvDecDecoder> cached_decoder;
    auto decoder = std::move(cached_decoder);
    if (!decoder || crop.left || crop.top || crop.right || crop.bottom) {
      decoder = std::make_unique<NvDecDecoder>();
    } else {
      decoder->reset();
    }
    decoder->init_decoder(config, codec, crop, width, height);
    auto nv12 = decoder->decode_packets(std::move(packets));
    auto result = std::make_shared<CUDAValue>(
        bgr ? nv12_to_planar_bgr(nv12, config, 1, false)
            : nv12_to_planar_rgb(nv12, config, 1, false));
    cached_decoder = std::move(decoder);
    return result;
  }
};
#endif
#endif
} // namespace

void register_fused(nb::module_& m) {
  // The factories are defined in every build, so that the stubs do not
  // depend on the build. They return the operation as a capsule, typed as
  // `object` so that the stubs do not depend on the Python version either.
  m.def(
      "_fused_transfer_buffer",
      [](const CUDAConfig& cfg [[maybe_unused]]) -> nb::object {
#ifdef SPDL_USE_CUDA
        return fused::to_capsule(std::make_shared<TransferOp>(cfg));
#else
        throw std::runtime_error("SPDL is not built with CUDA support");
#endif
      },
      nb::arg("device_config"));
  m.def(
      "_fused_nv12_convert",
      [](const CUDAConfig& cfg [[maybe_unused]],
         bool bgr [[maybe_unused]],
         int coeff [[maybe_unused]],
         bool sync [[maybe_unused]]) -> nb::object {
#ifdef SPDL_USE_CUDA
        return fused::to_capsule(
            std::make_shared<ColorOp>(cfg, bgr, coeff, sync));
#else
        throw std::runtime_error("SPDL is not built with CUDA support");
#endif
      },
      nb::arg("device_config"),
      nb::arg("bgr"),
      nb::arg("coeff"),
      nb::arg("sync"));
  m.def(
      "_fused_decode_nvdec",
      [](const CUDAConfig& cfg [[maybe_unused]],
         bool bgr [[maybe_unused]],
         int left [[maybe_unused]],
         int top [[maybe_unused]],
         int right [[maybe_unused]],
         int bottom [[maybe_unused]],
         int width [[maybe_unused]],
         int height [[maybe_unused]]) -> nb::object {
#if defined(SPDL_USE_CUDA) && defined(SPDL_USE_NVCODEC)
        return fused::to_capsule(
            std::make_shared<DecodeOp>(
                cfg,
                bgr,
                to_crop_area(left, top, right, bottom),
                width,
                height));
#else
        throw std::runtime_error("SPDL is not built with NVCODEC support.");
#endif
      },
      nb::arg("device_config"),
      nb::arg("bgr"),
      nb::arg("crop_left"),
      nb::arg("crop_top"),
      nb::arg("crop_right"),
      nb::arg("crop_bottom"),
      nb::arg("scale_width"),
      nb::arg("scale_height"));
}
} // namespace spdl::cuda
