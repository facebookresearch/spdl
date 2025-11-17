/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/cuda/transfer.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <fmt/core.h>

namespace nb = nanobind;

using cpu_array = nb::ndarray<nb::device::cpu, nb::c_contig>;
using cuda_array = nb::ndarray<nb::device::cuda, nb::c_contig>;

// Workaround for -Werror,-Wunused-variable in case SPDL_USE_CUDA
// is not defined. It hides the variable name.
#ifdef SPDL_USE_CUDA
#define _(var_name) var_name
#else
#define _(var_name)
#endif

namespace spdl::cuda {
using namespace spdl::core;
namespace {
#ifdef SPDL_USE_CUDA
ElemClass _get_elemclass(uint8_t code) {
  switch ((nb::dlpack::dtype_code)code) {
    case nb::dlpack::dtype_code::Int:
      return ElemClass::Int;
    case nb::dlpack::dtype_code::UInt:
      return ElemClass::UInt;
    case nb::dlpack::dtype_code::Float:
      return ElemClass::Float;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported DLPack type: {}", code));
  }
}
#endif
} // namespace

void register_transfer(nb::module_& m) {
  // CPU -> CUDA
  m.def(
      "transfer_buffer",
      [](CPUBufferPtr _(buffer), const CUDAConfig& _(cfg)) -> CUDABufferPtr {
#ifdef SPDL_USE_CUDA
        return transfer_buffer(std::move(buffer), cfg);
#else
        throw std::runtime_error("SPDL is not built with CUDA support");
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"));

  m.def(
      "transfer_buffer",
      [](cpu_array _(array), const CUDAConfig& _(cfg)) -> CUDABufferPtr {
#ifndef SPDL_USE_CUDA
        throw std::runtime_error("SPDL is not built with CUDA support");
#else
        std::vector<size_t> shape;
        auto src_ptr = array.shape_ptr();
        for (size_t i = 0; i < array.ndim(); ++i) {
          shape.push_back(src_ptr[i]);
        }
        return transfer_buffer(
            shape,
            _get_elemclass(array.dtype().code),
            array.itemsize(),
            array.data(),
            cfg);
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"));

  // CUDA -> CPU
  m.def(
      "transfer_buffer_cpu",
      [](cuda_array _(array)) -> spdl::core::CPUBufferPtr {
#ifndef SPDL_USE_CUDA
        throw std::runtime_error("SPDL is not built with CUDA support");
#else
        std::vector<size_t> shape;
        auto src_ptr = array.shape_ptr();
        for (size_t i = 0; i < array.ndim(); ++i) {
          shape.push_back(src_ptr[i]);
        }
        return transfer_buffer(
            shape,
            _get_elemclass(array.dtype().code),
            array.itemsize(),
            array.data());
#endif
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("buffer"));
}
} // namespace spdl::cuda
