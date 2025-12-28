/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/cuda/buffer.h>

#ifdef SPDL_USE_CUDA
#include <libspdl/cuda/color_conversion.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

namespace spdl::cuda {
void register_color_conversion(nb::module_& m) {
  m.def(
      "nv12_to_planar_rgb",
#ifndef SPDL_USE_CUDA
      [](const CUDABuffer&, const CUDAConfig&, int, bool) -> CUDABufferPtr {
        throw std::runtime_error("SPDL is not built with CUDA support.");
      },
#else
      &nv12_to_planar_rgb,
#endif
      R"(Convert batched NV12 frames to planar RGB.

Args:
    buffer: 3D buffer with shape ``[num_frames, height*1.5, width]``.
    device_config: The CUDA device configuration.
    matrix_coeff: Color matrix coefficients for conversion (default: ``BT.709``).
    sync: If ``True``, synchronizes the stream before returning.

Returns:
    CUDA buffer containing planar RGB data with shape ``[num_frames, 3, height, width]``.
)",
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("matrix_coeff") = 1,
      nb::arg("sync") = true);
  m.def(
      "nv12_to_planar_bgr",
#ifndef SPDL_USE_CUDA
      [](const CUDABuffer&, const CUDAConfig&, int, bool) -> CUDABufferPtr {
        throw std::runtime_error("SPDL is not built with CUDA support.");
      },
#else
      &nv12_to_planar_bgr,
#endif
      R"(Convert batched NV12 frames to planar BGR.

Args:
    buffer: 3D buffer with shape ``[num_frames, height*1.5, width]``.
    device_config: The CUDA device configuration.
    matrix_coeff: Color matrix coefficients for conversion (default: ``BT.709``).
    sync: If ``True``, synchronizes the stream before returning.

Returns:
    CUDA buffer containing planar BGR data with shape ``[num_frames, 3, height, width]``.
)",
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"),
      nb::arg("matrix_coeff") = 1,
      nb::arg("sync") = true);
}
} // namespace spdl::cuda
