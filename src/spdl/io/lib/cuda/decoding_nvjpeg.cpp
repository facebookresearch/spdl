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

#include "memoryview_utils.h"

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

void register_decoding_nvjpeg(nb::module_& m) {
  m.def(
      "decode_image_nvjpeg",
      [](const nb::memoryview& _(data),
         const CUDAConfig& _(cuda_config),
         int _(scale_width),
         int _(scale_height),
         const std::string& _(pix_fmt),
         bool _(sync)) -> CUDABufferPtr {
#ifndef SPDL_USE_NVJPEG
        NOT_SUPPORTED_NVJPEG;
#else
        return decode_image_nvjpeg(
            detail::memoryview_to_sv(data),
            cuda_config,
            scale_width,
            scale_height,
            pix_fmt,
            sync);
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width") = -1,
      nb::arg("scale_height") = -1,
      nb::arg("pix_fmt") = "rgb",
      nb::arg("sync") = true);

  m.def(
      "decode_image_nvjpeg",
      [](const std::vector<nb::memoryview>& _(data),
         const CUDAConfig& _(cuda_config),
         int _(scale_width),
         int _(scale_height),
         const std::string& _(pix_fmt),
         bool _(sync)) -> CUDABufferPtr {
#ifndef SPDL_USE_NVJPEG
        NOT_SUPPORTED_NVJPEG;
#else
        std::vector<std::string_view> dataset;
        dataset.reserve(data.size());
        for (const auto& d : data) {
          dataset.emplace_back(detail::memoryview_to_sv(d));
        }
        return decode_image_nvjpeg(
            dataset, cuda_config, scale_width, scale_height, pix_fmt, sync);
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("scale_width"),
      nb::arg("scale_height"),
      nb::arg("pix_fmt") = "rgb",
      nb::arg("sync") = true);
}
} // namespace spdl::cuda
