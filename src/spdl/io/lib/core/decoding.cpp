/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

#include "spdl_gil.h"

#include <cstring>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
FFmpegFramesPtr<media_type> decode(
    PacketsPtr<media_type>&& packets,
    const std::optional<DecodeConfig>& cfg,
    const std::optional<std::string>& filter_desc) {
  RELEASE_GIL();
  return decode_packets_ffmpeg(std::move(packets), cfg, filter_desc);
}

template <MediaType media_type>
DecoderPtr<media_type> _make_decoder(
    PacketsPtr<media_type>&& packets,
    const std::optional<DecodeConfig>& decode_cfg,
    const std::optional<std::string>& filter_desc) {
  RELEASE_GIL();
  return make_decoder(std::move(packets), decode_cfg, filter_desc);
}

template <MediaType media_type>
void _drop(DecoderPtr<media_type> decoder) {
  RELEASE_GIL();
  decoder.reset();
}

template <MediaType media_type>
std::optional<FFmpegFramesPtr<media_type>> _decode(
    StreamingDecoder<media_type>& decoder,
    int num_frames) {
  RELEASE_GIL();
  return decoder.decode(num_frames);
}

void zero_clear(nb::bytes data) {
  std::memset((void*)data.c_str(), 0, data.size());
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
  // Async decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_packets",
      &decode<MediaType::Audio>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());

  m.def(
      "decode_packets",
      &decode<MediaType::Video>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());

  m.def(
      "decode_packets",
      &decode<MediaType::Image>,
      nb::arg("packets"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("decode_config") = nb::none(),
      nb::arg("filter_desc") = nb::none());
}
} // namespace spdl::core
