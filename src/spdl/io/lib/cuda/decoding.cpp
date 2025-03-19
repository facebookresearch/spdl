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

using namespace spdl::core;

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  nb::class_<NvDecDecoder>(m, "NvDecDecoder")
#ifdef SPDL_USE_NVCODEC
      .def(
          "reset",
          &NvDecDecoder::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "set_init_flag",
          &NvDecDecoder::set_init_flag,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "decode",
          [](NvDecDecoder& self,
             PacketsPtr<MediaType::Video>&& packets,
             const CUDAConfig& cuda_config,
             int crop_left,
             int crop_top,
             int crop_right,
             int crop_bottom,
             int width,
             int height,
             const std::optional<std::string>& pix_fmt,
             bool flush) {
            return self.decode(
                std::move(packets),
                cuda_config,
                CropArea{
                    static_cast<short>(crop_left),
                    static_cast<short>(crop_top),
                    static_cast<short>(crop_right),
                    static_cast<short>(crop_bottom)},
                width,
                height,
                pix_fmt,
                flush);
          },
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          nb::kw_only(),
          nb::arg("device_config"),
          nb::arg("crop_left") = 0,
          nb::arg("crop_top") = 0,
          nb::arg("crop_right") = 0,
          nb::arg("crop_bottom") = 0,
          nb::arg("width") = -1,
          nb::arg("height") = -1,
          nb::arg("pix_fmt").none() = "rgb",
          nb::arg("flush") = false);
#endif
  ;

  m.def(
      "_nvdec_decoder",
      []() {
#ifdef SPDL_USE_NVCODEC
        return std::make_unique<NvDecDecoder>();
#else
        throw std::runtime_error("SPDL is not built with NVDEC support.");
#endif
      },
      nb::call_guard<nb::gil_scoped_release>());

  nb::class_<NvDecDecoder2>(m, "NvDecDecoder2")
#ifdef SPDL_USE_NVCODEC
      .def(
          "reset",
          &NvDecDecoder2::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "init",
          [](NvDecDecoder2& self,
             CUDAConfig cuda_config,
             spdl::core::VideoCodec codec,
             int crop_left,
             int crop_top,
             int crop_right,
             int crop_bottom,
             int width,
             int height) {
            self.init(
                std::move(cuda_config),
                std::move(codec),
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
          nb::arg("crop_left") = 0,
          nb::arg("crop_top") = 0,
          nb::arg("crop_right") = 0,
          nb::arg("crop_bottom") = 0,
          nb::arg("width") = -1,
          nb::arg("height") = -1)
      .def(
          "decode",
          &NvDecDecoder2::decode,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "flush",
          &NvDecDecoder2::flush,
          nb::call_guard<nb::gil_scoped_release>());
#endif
  ;

  m.def(
      "_nvdec_decoder2",
      []() {
#ifdef SPDL_USE_NVCODEC
        return std::make_unique<NvDecDecoder2>();
#else
        throw std::runtime_error("SPDL is not built with NVDEC support.");
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
