/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/cuda/nvdec/decoder.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::cuda {

#ifdef SPDL_USE_NVCODEC
#define _(var_name) var_name
#define __(impl) impl
#else
#define _NOT_SUPPORTED \
  throw std::runtime_error("SPDL is not built with NVCODEC support.")
#define _(var_name) var_name
#define __(impl) \
  { _NOT_SUPPORTED; }

NvDecDecoder::NvDecDecoder() {
  _NOT_SUPPORTED;
};
NvDecDecoder::~NvDecDecoder(){};
void NvDecDecoder::reset() {
  _NOT_SUPPORTED;
}
void NvDecDecoder::init(
    const CUDAConfig& cuda_config,
    const spdl::core::VideoCodec& codec,
    CropArea crop,
    int width,
    int height) {
  _NOT_SUPPORTED;
}
std::vector<CUDABuffer> NvDecDecoder::decode(
    spdl::core::VideoPacketsPtr packets) {
  _NOT_SUPPORTED;
}
std::vector<CUDABuffer> NvDecDecoder::flush() {
  _NOT_SUPPORTED;
}
#endif

using namespace spdl::core;

void register_decoding_nvdec(nb::module_& m) {
  nb::class_<NvDecDecoder>(m, "NvDecDecoder")
      .def(
          "reset",
          &NvDecDecoder::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "init",
          [](NvDecDecoder & _(self),
             const CUDAConfig& _(cuda_config),
             const spdl::core::VideoCodec& _(codec),
             int _(crop_left),
             int _(crop_top),
             int _(crop_right),
             int _(crop_bottom),
             int _(width),
             int _(height)) __({
            self.init(
                cuda_config,
                codec,
                CropArea{
                    static_cast<short>(crop_left),
                    static_cast<short>(crop_top),
                    static_cast<short>(crop_right),
                    static_cast<short>(crop_bottom)},
                width,
                height);
          }),
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("device_config"),
          nb::arg("codec"),
          nb::kw_only(),
          nb::arg("crop_left") = 0,
          nb::arg("crop_top") = 0,
          nb::arg("crop_right") = 0,
          nb::arg("crop_bottom") = 0,
          nb::arg("scale_width") = -1,
          nb::arg("scale_height") = -1)
      .def(
          "decode",
          &NvDecDecoder::decode,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def(
          "flush",
          &NvDecDecoder::flush,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_nvdec_decoder",
      []() -> std::unique_ptr<NvDecDecoder> __(
               { return std::make_unique<NvDecDecoder>(); }),
      nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::cuda
