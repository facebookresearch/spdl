/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/encoding.h>
#include <libspdl/cuda/transfer.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

using u8_cuda_array = nb::ndarray<uint8_t, nb::device::cuda, nb::c_contig>;

namespace spdl::cuda {
using namespace spdl::core;
namespace {

template <typename... Ts>
std::vector<size_t> get_shape(nb::ndarray<Ts...>& arr) {
  std::vector<size_t> ret;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    ret.push_back(arr.shape(i));
  }
  return ret;
}

void encode_cuda(
    std::string uri,
    u8_cuda_array data,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg) {
#ifndef SPDL_USE_CUDA
  throw std::runtime_error("SPDL is not built with CUDA,");
#else
  auto src = reinterpret_cast<void*>(data.data());
  auto shape = get_shape(data);
  if (data.dtype().code != (uint8_t)nb::dlpack::dtype_code::UInt) {
    throw std::runtime_error("Only unsigned interger type is supported");
  }
  auto depth = data.dtype().bits / 8;

  nb::gil_scoped_release release;

  auto storage = cp_to_cpu(src, shape);
  encode_image(uri, storage.data(), shape, depth, pix_fmt, encode_cfg);
#endif
}
} // namespace

void register_encoding(nb::module_& m) {
  m.def(
      "encode_image",
      &encode_cuda,
      nb::arg("path"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none());
}

} // namespace spdl::cuda
