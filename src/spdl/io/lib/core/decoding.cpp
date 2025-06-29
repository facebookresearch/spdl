/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

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

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media>
DecoderPtr<media> _make_decoder(
    const Codec<media>& codec,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc) {
  return std::make_unique<Decoder<media>>(codec, cfg, filter_desc);
}

template <MediaType media>
FramesPtr<media> decode_packets(
    PacketsPtr<media> packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc,
    int num_frames) {
  if (!packets->codec) {
    throw std::runtime_error("Packets does not have codec info.");
  }
  Decoder<media> decoder{*packets->codec, cfg, filter_desc};
  return decoder.decode_and_flush(std::move(packets), num_frames);
}

template <MediaType media>
FramesPtr<media> decode(Decoder<media>& self, PacketsPtr<media> packets) {
  return self.decode(std::move(packets), false, -1);
}
} // namespace

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Decoder
  ////////////////////////////////////////////////////////////////////////////////
  nb::class_<Decoder<MediaType::Audio>>(m, "AudioDecoder")
      .def(
          "decode",
          &Decoder<MediaType::Audio>::decode,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def("flush", &Decoder<MediaType::Audio>::flush);
  nb::class_<Decoder<MediaType::Video>>(m, "VideoDecoder")
      .def(
          "decode",
          &Decoder<MediaType::Video>::decode,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def("flush", &Decoder<MediaType::Video>::flush);
  nb::class_<Decoder<MediaType::Image>>(m, "ImageDecoder")
      .def(
          "decode",
          &Decoder<MediaType::Image>::decode,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def("flush", &Decoder<MediaType::Image>::flush);

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
