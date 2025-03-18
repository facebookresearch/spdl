/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::cuda {

void register_utils(nb::module_& m) {
  m.def(
      "is_cuda_available",
      []() {
        return
#ifdef SPDL_USE_CUDA
            true
#else
            false
#endif
            ;
      },
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "is_nvcodec_available",
      []() {
        return
#ifdef SPDL_USE_NVCODEC
            true
#else
            false
#endif
            ;
      },
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "is_nvjpeg_available",
      []() {
        return
#ifdef SPDL_USE_NVJPEG
            true
#else
            false
#endif
            ;
      },
      nb::call_guard<nb::gil_scoped_release>());
}

} // namespace spdl::cuda
