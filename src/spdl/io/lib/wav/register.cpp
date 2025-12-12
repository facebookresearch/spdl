/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_utils.h"

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {
NB_MODULE(_wav, m) {
  nb::class_<WAVHeader>(m, "WAVHeader", "WAV file header information.")
      .def_ro(
          "audio_format",
          &WAVHeader::audio_format,
          "Audio format code (1=PCM, 3=IEEE float, etc.)")
      .def_ro(
          "num_channels", &WAVHeader::num_channels, "Number of audio channels")
      .def_ro("sample_rate", &WAVHeader::sample_rate, "Sample rate in Hz")
      .def_ro(
          "byte_rate",
          &WAVHeader::byte_rate,
          "Byte rate (sample_rate * num_channels * bits_per_sample / 8)")
      .def_ro(
          "block_align",
          &WAVHeader::block_align,
          "Block alignment (num_channels * bits_per_sample / 8)")
      .def_ro("bits_per_sample", &WAVHeader::bits_per_sample, "Bits per sample")
      .def_ro(
          "data_size", &WAVHeader::data_size, "Size of audio data in bytes");
  m.def(
      "load_wav",
      [](const nb::bytes& data,
         std::optional<double> time_offset_seconds = std::nullopt,
         std::optional<double> duration_seconds = std::nullopt) -> nb::dict {
        const char* p = data.c_str();
        size_t s = data.size();
        WAVHeader header;
        size_t num_samples;
        std::string_view view;
        {
          nb::gil_scoped_release _{};
          std::string_view d{p, s};
          header = parse_wav_header(d);
          view = extract_wav_samples(d, time_offset_seconds, duration_seconds);
          num_samples = view.size() / header.block_align;
        }

        nb::dict array_interface;
        array_interface["version"] = 3;
        array_interface["shape"] =
            nb::make_tuple(num_samples, header.num_channels);

        nb::str typestr = [&header]() -> nb::str {
          switch (header.bits_per_sample) {
            case 8:
              return nb::str{"|u1"};
            case 16:
              return nb::str{"<i2"};
            case 32: {
              return nb::str{header.audio_format == 3 ? "<f4" : "<i4"};
            }
            case 64:
              return nb::str{"<f8"};
            default:
              throw std::domain_error(
                  "Unsupported bits_per_sample: " +
                  std::to_string(header.bits_per_sample));
          }
        }();
        array_interface["typestr"] = typestr;
        array_interface["data"] =
            nb::make_tuple(reinterpret_cast<uintptr_t>(view.data()), false);
        array_interface["owner"] = data;

        return array_interface;
      },
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("time_offset_seconds") = nb::none(),
      nb::arg("duration_seconds") = nb::none(),
      "Extract audio samples from WAV data.\n\n"
      "Args:\n"
      "    wav_data: Binary WAV data as bytes or string\n"
      "    time_offset_seconds: Optional starting time in seconds (default: 0.0)\n"
      "    duration_seconds: Optional duration in seconds (default: until end)\n\n"
      "Returns:\n"
      "    dict: Dictionary compliant with Array Interface Protocol containing:\n"
      "        - version: Protocol version (3)\n"
      "        - shape: Tuple of (num_samples, num_channels)\n"
      "        - typestr: Data type string\n"
      "        - data: Tuple of (data_pointer, read_only_flag)\n"
      "        - owner: Object owning the data buffer\n\n"
      "Raises:\n"
      "    WAVParseError: If the WAV data is invalid or time range is out of bounds");

  m.def(
      "parse_wav",
      [](const nb::bytes& data) -> WAVHeader {
        const char* p = data.c_str();
        size_t s = data.size();
        WAVHeader header;
        {
          nb::gil_scoped_release _{};
          std::string_view d{p, s};
          header = parse_wav_header(d);
        }
        return header;
      },
      nb::arg("data"),
      "Parse WAV file header and extract metadata.\n\n"
      "Args:\n"
      "    data: Binary WAV data as bytes\n\n"
      "Returns:\n"
      "    WAVHeader: Object containing WAV header information with attributes:\n"
      "        - audio_format: Audio format code (1=PCM, 3=IEEE float, etc.)\n"
      "        - num_channels: Number of audio channels\n"
      "        - sample_rate: Sample rate in Hz\n"
      "        - byte_rate: Byte rate (sample_rate * num_channels * bits_per_sample / 8)\n"
      "        - block_align: Block alignment (num_channels * bits_per_sample / 8)\n"
      "        - bits_per_sample: Bits per sample\n"
      "        - data_size: Size of audio data in bytes\n\n"
      "Raises:\n"
      "    WAVParseError: If the WAV data is invalid or malformed");
}
} // namespace
} // namespace spdl::core
