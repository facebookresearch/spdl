/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace spdl::core {

struct WAVHeader {
  uint16_t audio_format;
  uint16_t num_channels;
  uint32_t sample_rate;
  uint32_t byte_rate;
  uint16_t block_align;
  uint16_t bits_per_sample;
  uint32_t data_size;
};

std::string_view extract_wav_samples(
    std::string_view wav_data,
    std::optional<double> time_offset_seconds = std::nullopt,
    std::optional<double> duration_seconds = std::nullopt);

WAVHeader parse_wav_header(std::string_view wav_data);

} // namespace spdl::core
