/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/encoder.h>
#include <libspdl/core/encoding.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

using cpu_array = nb::ndarray<nb::device::cpu, nb::c_contig>;

namespace spdl::core {
namespace {

template <typename... Ts>
std::vector<size_t> get_shape(nb::ndarray<Ts...>& arr) {
  std::vector<size_t> ret;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    ret.push_back(arr.shape(i));
  }
  return ret;
}

void encode(
    std::string path,
    cpu_array data,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg) {
  auto src = reinterpret_cast<void*>(data.data());
  auto shape = get_shape(data);
  if (data.dtype().code != (uint8_t)nb::dlpack::dtype_code::UInt) {
    throw std::runtime_error("Only unsigned interger type is supported");
  }
  auto depth = data.dtype().bits / 8;
  nb::gil_scoped_release __g;
  encode_image(path, src, shape, depth, pix_fmt, encode_cfg);
}

} // namespace

void register_encoding(nb::module_& m) {
  nb::class_<VideoEncodeConfig>(m, "VideoEncodeConfig");

  m.def(
      "video_encode_config",
      [](int height,
         int width,
         const std::optional<std::tuple<int, int>> frame_rate,
         const std::optional<std::string>& pix_fmt,
         int bit_rate,
         int compression_level,
         int qscale,
         int gop_size,
         int max_b_frames) {
        auto fr = [&]() -> std::optional<Rational> {
          if (frame_rate) {
            auto rate = *frame_rate;
            return Rational{std::get<0>(rate), std::get<1>(rate)};
          }
          return std::nullopt;
        }();
        return VideoEncodeConfig{
            height,
            width,
            pix_fmt,
            fr,
            bit_rate,
            compression_level,
            qscale,
            gop_size,
            max_b_frames};
      },
      nb::call_guard<nb::gil_scoped_release>(),
      nb::kw_only(),
      nb::arg("height"),
      nb::arg("width"),
      nb::arg("frame_rate") = std::nullopt,
      nb::arg("pix_fmt") = std::nullopt,
      nb::arg("bit_rate") = -1,
      nb::arg("compression_level") = -1,
      nb::arg("qscale") = -1,
      nb::arg("gop_size") = -1,
      nb::arg("max_b_frames") = -1);

  nb::class_<VideoEncoder>(m, "VideoEncoder")
      .def(
          "encode",
          &VideoEncoder::encode,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "flush",
          &VideoEncoder::flush,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "encode_image",
      &encode,
      nb::arg("path"),
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none());
}

} // namespace spdl::core
