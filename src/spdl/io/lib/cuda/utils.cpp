/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/utils.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::cuda {

void register_utils(nb::module_& m) {
  m.def(
      "init",
#ifdef SPDL_USE_CUDA
      init_cuda,
#else
      []() {
        throw std::runtime_error("SPDL is not built with CUDA support.");
      },
#endif
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "built_with_cuda",
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
      "built_with_nvcodec",
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
      "built_with_nvjpeg",
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

  m.def(
      "synchronize_stream",
#ifdef SPDL_USE_CUDA
      synchronize_stream,
#else
      [](nb::object) {
        throw std::runtime_error("SPDL is not built with CUDA support.");
      },
#endif
      nb::call_guard<nb::gil_scoped_release>());
}

} // namespace spdl::cuda
