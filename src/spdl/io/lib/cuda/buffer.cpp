/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/core/buffer.h>
#include <libspdl/cuda/buffer.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Workaround for -Werror,-Wunused-variable in case SPDL_USE_CUDA
// is not defined. It hides the variable name.
#ifdef SPDL_USE_CUDA
#define _(var_name) var_name
#else
#define _(var_name)
#endif

namespace spdl::cuda {
using namespace spdl::core;
#ifdef SPDL_USE_CUDA
namespace {
////////////////////////////////////////////////////////////////////////////////
// Array interface supplements
////////////////////////////////////////////////////////////////////////////////
std::string get_typestr(const ElemClass elem_class, size_t depth) {
  const auto key = [&]() {
    switch (elem_class) {
      case ElemClass::UInt:
        return "u";
      case ElemClass::Int:
        return "i";
      case ElemClass::Float:
        return "f";
      default:
        throw std::runtime_error(
            fmt::format("Unsupported class {}.", int(elem_class)));
    }
  }();
  return fmt::format("|{}{}", key, depth);
}

nb::dict get_cuda_array_interface(CUDABuffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
  nb::dict ret;
  ret["version"] = 2;
  ret["shape"] = nb::tuple(nb::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = nb::none();
  ret["stream"] = b.get_cuda_stream();
  return ret;
}
} // namespace
#endif

void register_buffers(nb::module_& m) {
  nb::class_<CUDABuffer>(m, "CUDABuffer")
      .def_prop_ro(
          "__cuda_array_interface__",
          [](CUDABuffer& _(self)) -> nb::dict {
#ifndef SPDL_USE_CUDA
            throw std::runtime_error("SPDL is not compiled with CUDA support.");
#else
            return get_cuda_array_interface(self);
#endif
          })
      .def_prop_ro(
          "device_index",
          [](CUDABuffer& _(self)) -> int {
#ifndef SPDL_USE_CUDA
            throw std::runtime_error("SPDL is not compiled with CUDA support.");
#else
            return self.device_index;
#endif
          },
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::cuda
