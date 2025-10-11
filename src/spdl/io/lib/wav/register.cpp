/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {
NB_MODULE(_wav, m) {
  m.def(
      "parse_wav_header",
      [](nb::bytes data) {
        std::string_view data_{data.c_str(), data.size()};
        WAVHeader header = parse_wav_header(data_);
        return nb::make_tuple(
            header.audio_format,
            header.num_channels,
            header.sample_rate,
            header.byte_rate,
            header.block_align,
            header.bits_per_sample,
            header.data_size);
      },
      nb::arg("wav_data"),
      "Parse WAV header from audio data.\n\n"
      "Args:\n"
      "    wav_data: Binary WAV data as bytes or string\n\n"
      "Returns:\n"
      "    tuple: (audio_format, num_channels, sample_rate, byte_rate, "
      "block_align, bits_per_sample, data_size)\n\n"
      "Raises:\n"
      "    WAVParseError: If the WAV data is invalid or cannot be parsed");

  m.def(
      "load_wav",
      [](nb::bytes data,
         std::optional<double> time_offset_seconds = std::nullopt,
         std::optional<double> duration_seconds = std::nullopt) -> nb::object {
        const char* p = data.c_str();
        size_t s = data.size();
        WAVHeader header;
        std::string_view view;
        {
          nb::gil_scoped_release _{};
          std::string_view data_{p, s};
          header = parse_wav_header(data_);
          view =
              extract_wav_samples(data_, time_offset_seconds, duration_seconds);
        }
        size_t num_samples = view.size() / header.block_align;
        size_t shape[2] = {num_samples, header.num_channels};
        if (header.bits_per_sample == 8) {
          return nb::cast(nb::ndarray<nb::numpy, uint8_t>{
              const_cast<uint8_t*>(
                  reinterpret_cast<const uint8_t*>(view.data())),
              2,
              shape,
              data.ptr()});
        } else if (header.bits_per_sample == 16) {
          return nb::cast(nb::ndarray<nb::numpy, int16_t>{
              const_cast<int16_t*>(
                  reinterpret_cast<const int16_t*>(view.data())),
              2,
              shape,
              data.ptr()});
        } else if (header.bits_per_sample == 32) {
          if (header.audio_format == 3) {
            // IEEE float
            return nb::cast(nb::ndarray<nb::numpy, float>{
                const_cast<float*>(reinterpret_cast<const float*>(view.data())),
                2,
                shape,
                data.ptr()});
          } else {
            return nb::cast(nb::ndarray<nb::numpy, int32_t>{
                const_cast<int32_t*>(
                    reinterpret_cast<const int32_t*>(view.data())),
                2,
                shape,
                data.ptr()});
          }
        } else if (header.bits_per_sample == 64) {
          return nb::cast(nb::ndarray<nb::numpy, double>{
              const_cast<double*>(reinterpret_cast<const double*>(view.data())),
              2,
              shape,
              data.ptr()});
        } else {
          throw std::domain_error(
              "Unsupported bits_per_sample: " +
              std::to_string(header.bits_per_sample));
        }
      },
      nb::arg("wav_data"),
      nb::kw_only(),
      nb::arg("time_offset_seconds") = nb::none(),
      nb::arg("duration_seconds") = nb::none(),
      "Extract audio samples from WAV data as numpy array.\n\n"
      "Args:\n"
      "    wav_data: Binary WAV data as bytes or string\n"
      "    time_offset_seconds: Optional starting time in seconds (default: 0.0)\n"
      "    duration_seconds: Optional duration in seconds (default: until end)\n\n"
      "Returns:\n"
      "    ndarray: Audio samples as numpy array with shape (num_samples, num_channels)\n"
      "             The dtype depends on bits_per_sample:\n"
      "             - 8 bits: uint8\n"
      "             - 16 bits: int16\n"
      "             - 32 bits: int32 or float32\n"
      "             - 64 bits: float64\n\n"
      "Raises:\n"
      "    WAVParseError: If the WAV data is invalid or time range is out of bounds");
}
} // namespace
} // namespace spdl::core
