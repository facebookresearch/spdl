/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/transfer.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <fmt/core.h>

#include "spdl_gil.h"

namespace nb = nanobind;

using cpu_array = nb::ndarray<nb::device::cpu, nb::c_contig>;
using cuda_array = nb::ndarray<nb::device::cuda, nb::c_contig>;

namespace spdl::core {
namespace {
CUDABufferPtr _transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg) {
  RELEASE_GIL();
  return transfer_buffer(std::move(buffer), cfg);
}

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

CUDABufferPtr _transfer_cpu_array(cpu_array array, const CUDAConfig& cfg) {
  RELEASE_GIL();
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
}

CPUBufferPtr _transfer_cuda_array(cuda_array array) {
  RELEASE_GIL();
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
}

} // namespace

void register_transfer(nb::module_& m) {
  // CPU -> CUDA
  m.def(
      "transfer_buffer",
      &_transfer_buffer,
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"));

  m.def(
      "transfer_buffer",
      &_transfer_cpu_array,
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("device_config"));

  // CUDA -> CPU
  m.def("transfer_buffer_cpu", &_transfer_cuda_array, nb::arg("buffer"));
}
} // namespace spdl::core
