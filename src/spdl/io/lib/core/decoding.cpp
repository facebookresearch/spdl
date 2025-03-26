/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/decoder.h>
#include <libspdl/core/decoding.h>

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
StreamingDecoderPtr<media_type> _make_decoder(
    PacketsPtr<media_type>&& packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::optional<std::string>& filter_desc) {
  nb::gil_scoped_release __g;
  return make_decoder(std::move(packets), decode_cfg, filter_desc);
}

template <MediaType media_type>
void _drop(StreamingDecoderPtr<media_type> decoder) {
  nb::gil_scoped_release __g;
  decoder.reset();
}

template <MediaType media_type>
std::optional<FFmpegFramesPtr<media_type>> _decode(
    StreamingDecoder<media_type>& decoder,
    int num_frames) {
  nb::gil_scoped_release __g;
  return decoder.decode(num_frames);
}

void zero_clear(nb::bytes data) {
  std::memset((void*)data.c_str(), 0, data.size());
}

template <MediaType media_type>
DecoderPtr<media_type> make_decoder_(
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
  return decoder.decode(std::move(packets), num_frames);
}

} // namespace

void register_decoding(nb::module_& m) {
  nb::class_<StreamingDecoder<MediaType::Video>>(m, "StreamingVideoDecoder")
      .def("decode", &_decode<MediaType::Video>);

  m.def(
      "_streaming_decoder",
      &_make_decoder<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());

  m.def("_drop", &_drop<MediaType::Video>);

  ////////////////////////////////////////////////////////////////////////////////
  // Decoder
  ////////////////////////////////////////////////////////////////////////////////
  nb::class_<Decoder<MediaType::Audio>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Audio>::decode,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Decoder<MediaType::Video>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Video>::decode,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Decoder<MediaType::Image>>(m, "Decoder")
      .def(
          "decode",
          &Decoder<MediaType::Image>::decode,
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_make_decoder",
      &make_decoder_<MediaType::Audio>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("codec"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());
  m.def(
      "_make_decoder",
      &make_decoder_<MediaType::Video>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("codec"),
      nb::kw_only(),
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());
  m.def(
      "_make_decoder",
      &make_decoder_<MediaType::Image>,
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
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);

  m.def(
      "decode_packets",
      &decode_packets<MediaType::Video>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);

  m.def(
      "decode_packets",
      &decode_packets<MediaType::Image>,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none(),
      nb::arg("num_frames") = -1);
}
} // namespace spdl::core
