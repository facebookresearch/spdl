/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/nvdec/decoder.h>
#include <libspdl/cuda/nvjpeg/decoding.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::cuda {

#define NOT_SUPPORTED \
  throw std::runtime_error("SPDL is not built with NVCODEC support.")

#ifndef SPDL_USE_NVCODEC
NvDecDecoder::NvDecDecoder(){};
NvDecDecoder::~NvDecDecoder(){};
void NvDecDecoder::reset() {
  NOT_SUPPORTED;
}
void NvDecDecoder::init(
    const CUDAConfig& cuda_config,
    const spdl::core::VideoCodec& codec,
    CropArea crop,
    int width,
    int height) {
  NOT_SUPPORTED;
}
std::vector<CUDABuffer> NvDecDecoder::decode(
    spdl::core::VideoPacketsPtr packets) {
  NOT_SUPPORTED;
}
std::vector<CUDABuffer> NvDecDecoder::flush() {
  NOT_SUPPORTED;
}
#endif

using namespace spdl::core;

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  nb::class_<NvDecDecoder>(m, "NvDecDecoder")
      .def(
          "reset",
          &NvDecDecoder::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "init",
          [](NvDecDecoder& self,
             const CUDAConfig& cuda_config,
             const spdl::core::VideoCodec& codec,
             int crop_left,
             int crop_top,
             int crop_right,
             int crop_bottom,
             int width,
             int height) {
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
          },
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
      []() -> std::unique_ptr<NvDecDecoder> {
#ifdef SPDL_USE_NVCODEC
        return std::make_unique<NvDecDecoder>();
#else
        NOT_SUPPORTED;
#endif
      },
      nb::call_guard<nb::gil_scoped_release>());

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVJPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_image_nvjpeg",
      [](nb::bytes data,
         const CUDAConfig& cuda_config,
         int scale_width,
         int scale_height,
         const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
        throw std::runtime_error("SPDL is not built with NVJPEG support.");
#else
        auto ret = decode_image_nvjpeg(
            std::string_view{data.c_str(), data.size()},
            cuda_config,
            scale_width,
            scale_height,
            pix_fmt);
        return ret;
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width") = -1,
      nb::arg("scale_height") = -1,
      nb::arg("pix_fmt") = "rgb");

  m.def(
      "decode_image_nvjpeg",
      [](const std::vector<nb::bytes>& data,
         const CUDAConfig& cuda_config,
         int scale_width,
         int scale_height,
         const std::string& pix_fmt) {
#ifndef SPDL_USE_NVJPEG
        throw std::runtime_error("SPDL is not built with NVJPEG support.");
#else
        std::vector<std::string_view> dataset;
        for (const auto& d : data) {
          dataset.push_back(std::string_view{d.c_str(), d.size()});
        }
        return decode_image_nvjpeg(
            dataset, cuda_config, scale_width, scale_height, pix_fmt);
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width"),
      nb::arg("scale_height"),
      nb::arg("pix_fmt") = "rgb");
}
} // namespace spdl::cuda
