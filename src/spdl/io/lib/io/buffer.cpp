/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/buffer.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "spdl_gil.h"

namespace nb = nanobind;

namespace spdl::core {
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

nb::dict get_array_interface(CPUBuffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
  nb::dict ret;
  ret["version"] = 3;
  ret["shape"] = nb::tuple(nb::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = nb::none();
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", typestr}};
  return ret;
}

#ifdef SPDL_USE_CUDA
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
#endif

} // namespace

void register_buffers(nb::module_& m) {
#ifdef SPDL_USE_CUDA
#define IF_CUDABUFFER_ENABLED(x) x
#else
#define IF_CUDABUFFER_ENABLED(x)                                         \
  [](const CUDABuffer&) {                                                \
    throw std::runtime_error("SPDL is not compiled with CUDA support."); \
  }
#endif

  nb::class_<CPUBuffer>(m, "CPUBuffer")
      .def_prop_ro("__array_interface__", [](CPUBuffer& self) {
        return get_array_interface(self);
      });

  nb::class_<CUDABuffer>(m, "CUDABuffer")
      .def_prop_ro(
          "__cuda_array_interface__",
          IF_CUDABUFFER_ENABLED(
              [](CUDABuffer& self) { return get_cuda_array_interface(self); }))
      .def_prop_ro("device_index", IF_CUDABUFFER_ENABLED([](CUDABuffer& self) {
                     RELEASE_GIL();
                     return self.device_index;
                   }));
}
} // namespace spdl::core
