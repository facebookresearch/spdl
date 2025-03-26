/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoder.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
DecoderPtr<media_type> _make_decoder(
    const Codec<media_type>& codec,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc) {
  return std::make_unique<Decoder<media_type>>(codec, cfg, filter_desc);
}

template <MediaType media_type>
FFmpegFramesPtr<media_type> decode_packets(
    PacketsPtr<media_type> packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc,
    int num_frames) {
  Decoder<media_type> decoder{packets->codec, cfg, filter_desc};
  return decoder.decode_and_flush(std::move(packets), num_frames);
}

} // namespace

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Decoder
  ////////////////////////////////////////////////////////////////////////////////
  nb::class_<Decoder<MediaType::Audio>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Audio>::decode_and_flush,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Decoder<MediaType::Video>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Video>::decode_and_flush,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Decoder<MediaType::Image>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Image>::decode_and_flush,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_make_decoder",
      &_make_decoder<MediaType::Audio>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("codec"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());
  m.def(
      "_make_decoder",
      &_make_decoder<MediaType::Video>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("codec"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());
  m.def(
      "_make_decoder",
      &_make_decoder<MediaType::Image>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("codec"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());

  ////////////////////////////////////////////////////////////////////////////////
  // Decoding
  ////////////////////////////////////////////////////////////////////////////////

  m.def(
      "decode_packets",
      &decode_packets<MediaType::Audio>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);

  m.def(
      "decode_packets",
      &decode_packets<MediaType::Video>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);

  m.def(
      "decode_packets",
      &decode_packets<MediaType::Image>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);
}
} // namespace spdl::core
