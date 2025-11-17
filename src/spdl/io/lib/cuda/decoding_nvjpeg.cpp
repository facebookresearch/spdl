/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/cuda/nvjpeg/decoding.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::cuda {

#define NOT_SUPPORTED_NVJPEG \
  throw std::runtime_error("SPDL is not built with NVJPEG support.")

// Workaround for -Werror,-Wunused-variable in case SPDL_USE_NVJPEG
// is not defined. It hides the variable name.
#ifdef SPDL_USE_NVJPEG
#define _(var_name) var_name
#else
#define _(var_name)
#endif

using namespace spdl::core;

namespace {
#ifdef SPDL_USE_NVJPEG
void zero_clear(const nb::bytes& data) {
  std::memset((void*)data.c_str(), 0, data.size());
}
#endif
} // namespace

void register_decoding_nvjpeg(nb::module_& m) {
  m.def(
      "decode_image_nvjpeg",
      [](nb::bytes _(data),
         const CUDAConfig& _(cuda_config),
         int _(scale_width),
         int _(scale_height),
         const std::string& _(pix_fmt),
         bool _(_zero_clear)) -> CUDABufferPtr {
#ifndef SPDL_USE_NVJPEG
        NOT_SUPPORTED_NVJPEG;
#else
        auto ret = decode_image_nvjpeg(
            std::string_view{data.c_str(), data.size()},
            cuda_config,
            scale_width,
            scale_height,
            pix_fmt);
        if (_zero_clear) {
          zero_clear(data);
        }
        return ret;
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width") = -1,
      nb::arg("scale_height") = -1,
      nb::arg("pix_fmt") = "rgb",
      nb::arg("_zero_clear") = false);

  m.def(
      "decode_image_nvjpeg",
      [](const std::vector<nb::bytes>& _(data),
         const CUDAConfig& _(cuda_config),
         int _(scale_width),
         int _(scale_height),
         const std::string& _(pix_fmt),
         bool _(_zero_clear)) -> CUDABufferPtr {
#ifndef SPDL_USE_NVJPEG
        NOT_SUPPORTED_NVJPEG;
#else
        std::vector<std::string_view> dataset;
        dataset.reserve(data.size());
        for (const auto& d : data) {
          dataset.emplace_back(d.c_str(), d.size());
        }
        auto ret = decode_image_nvjpeg(
            dataset, cuda_config, scale_width, scale_height, pix_fmt);
        if (_zero_clear) {
          for (const auto& d : data) {
            zero_clear(d);
          }
        }
        return ret;
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width"),
      nb::arg("scale_height"),
      nb::arg("pix_fmt") = "rgb",
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::cuda
