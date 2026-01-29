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
  return decoder.decode_packets(std::move(packets), num_frames);
}
} // namespace

void register_decoding(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Decoder
  ////////////////////////////////////////////////////////////////////////////////
  using AudioFramesGenerator = Generator<AudioFramesPtr>;
  using VideoFramesGenerator = Generator<VideoFramesPtr>;
  using ImageFramesGenerator = Generator<ImageFramesPtr>;

  nb::class_<Decoder<MediaType::Audio>>(
      m,
      "AudioDecoder",
      "Decode stream of audio packets. See :py:class:`Decoder` for the detail.")
      .def(
          "streaming_decode_packets",
          &Decoder<MediaType::Audio>::streaming_decode_packets,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          "Streaming decode packets and yield frames")
      .def(
          "flush",
          &Decoder<MediaType::Audio>::flush,
          nb::call_guard<nb::gil_scoped_release>(),
          "Flush the decoder and yield remaining frames");

  nb::class_<Decoder<MediaType::Video>>(
      m,
      "VideoDecoder",
      "Decode stream of video packets. See :py:class:`Decoder` for the detail.")
      .def(
          "streaming_decode_packets",
          &Decoder<MediaType::Video>::streaming_decode_packets,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"),
          "Streaming decode packets and yield frames")
      .def(
          "flush",
          &Decoder<MediaType::Video>::flush,
          nb::call_guard<nb::gil_scoped_release>(),
          "Flush the decoder and yield remaining frames");

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

  nb::class_<AudioFramesGenerator>(m, "AudioFramesIterator")
      .def(
          "__iter__",
          [](AudioFramesGenerator& self) -> AudioFramesGenerator& {
            return self;
          },
          nb::rv_policy::reference)
      .def(
          "__next__",
          [](AudioFramesGenerator& self) -> AudioFramesPtr {
            if (!self) {
              throw nb::stop_iteration();
            }
            return self();
          },
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<VideoFramesGenerator>(m, "VideoFramesIterator")
      .def(
          "__iter__",
          [](VideoFramesGenerator& self) -> VideoFramesGenerator& {
            return self;
          },
          nb::rv_policy::reference)
      .def(
          "__next__",
          [](VideoFramesGenerator& self) -> VideoFramesPtr {
            if (!self) {
              throw nb::stop_iteration();
            }
            return self();
          },
          nb::call_guard<nb::gil_scoped_release>());

  nb::class_<ImageFramesGenerator>(m, "ImageFramesIterator")
      .def(
          "__iter__",
          [](ImageFramesGenerator& self) -> ImageFramesGenerator& {
            return self;
          },
          nb::rv_policy::reference)
      .def(
          "__next__",
          [](ImageFramesGenerator& self) -> ImageFramesPtr {
            if (!self) {
              throw nb::stop_iteration();
            }
            return self();
          },
          nb::call_guard<nb::gil_scoped_release>());

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
