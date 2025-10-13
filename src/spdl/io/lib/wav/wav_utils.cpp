/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace spdl::core {

namespace {

template <typename T>
T read_little_endian(const char* data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

bool check_fourcc(std::string_view data, size_t offset, const char* expected) {
  if (offset + 4 > data.size()) {
    return false;
  }
  return std::memcmp(data.data() + offset, expected, 4) == 0;
}

} // namespace

WAVHeader parse_wav_header(std::string_view wav_data) {
  if (wav_data.size() < 44) {
    throw std::domain_error("WAV data too small to contain valid header");
  }

  // Verify RIFF header
  if (!check_fourcc(wav_data, 0, "RIFF")) {
    throw std::domain_error("Missing RIFF header");
  }

  // Verify WAVE format
  if (!check_fourcc(wav_data, 8, "WAVE")) {
    throw std::domain_error("Missing WAVE format identifier");
  }

  // Find and parse fmt chunk
  size_t offset = 12;
  bool found_fmt = false;
  WAVHeader header{};

  while (offset + 8 <= wav_data.size()) {
    if (check_fourcc(wav_data, offset, "fmt ")) {
      uint32_t fmt_size =
          read_little_endian<uint32_t>(wav_data.data() + offset + 4);

      if (offset + 8 + fmt_size > wav_data.size()) {
        throw std::domain_error("fmt chunk extends beyond file size");
      }

      if (fmt_size < 16) {
        throw std::domain_error("fmt chunk too small");
      }

      const char* fmt_data = wav_data.data() + offset + 8;
      header.audio_format = read_little_endian<uint16_t>(fmt_data);
      header.num_channels = read_little_endian<uint16_t>(fmt_data + 2);
      header.sample_rate = read_little_endian<uint32_t>(fmt_data + 4);
      header.byte_rate = read_little_endian<uint32_t>(fmt_data + 8);
      header.block_align = read_little_endian<uint16_t>(fmt_data + 12);
      header.bits_per_sample = read_little_endian<uint16_t>(fmt_data + 14);

      found_fmt = true;
      offset += 8 + fmt_size;
      break;
    }
    offset += 4;
  }

  if (!found_fmt) {
    throw std::domain_error("fmt chunk not found");
  }

  // Find data chunk
  while (offset + 8 <= wav_data.size()) {
    if (check_fourcc(wav_data, offset, "data")) {
      header.data_size =
          read_little_endian<uint32_t>(wav_data.data() + offset + 4);
      return header;
    }

    // Skip this chunk
    if (offset + 8 <= wav_data.size()) {
      uint32_t chunk_size =
          read_little_endian<uint32_t>(wav_data.data() + offset + 4);
      offset += 8 + chunk_size;
    } else {
      break;
    }
  }

  throw std::domain_error("data chunk not found");
}

std::string_view extract_wav_samples(
    std::string_view wav_data,
    std::optional<double> time_offset_seconds,
    std::optional<double> duration_seconds) {
  // Parse the WAV header
  WAVHeader header = parse_wav_header(wav_data);

  // Validate header
  if (header.num_channels == 0) {
    throw std::domain_error("Invalid number of channels: 0");
  }
  if (header.sample_rate == 0) {
    throw std::domain_error("Invalid sample rate: 0");
  }
  if (header.block_align == 0) {
    throw std::domain_error("Invalid block align: 0");
  }

  // Find the start of the data chunk
  size_t offset = 12;
  size_t data_offset = 0;

  while (offset + 8 <= wav_data.size()) {
    if (check_fourcc(wav_data, offset, "data")) {
      data_offset = offset + 8;
      break;
    }

    // Skip this chunk
    uint32_t chunk_size =
        read_little_endian<uint32_t>(wav_data.data() + offset + 4);
    offset += 8 + chunk_size;
  }

  if (data_offset == 0) {
    throw std::domain_error("data chunk not found");
  }

  // Calculate available data size
  size_t available_data_size = std::min(
      static_cast<size_t>(header.data_size), wav_data.size() - data_offset);

  // If no time window specified, return entire waveform
  if (!time_offset_seconds && !duration_seconds) {
    return wav_data.substr(data_offset, available_data_size);
  }

  // Calculate byte offsets for time window
  double offset_sec = time_offset_seconds.value_or(0.0);
  if (offset_sec < 0.0) {
    throw std::domain_error("Time offset cannot be negative");
  }

  // Calculate starting sample and byte offset
  uint64_t start_sample =
      static_cast<uint64_t>(offset_sec * header.sample_rate);
  uint64_t start_byte = start_sample * header.block_align;

  if (start_byte >= available_data_size) {
    throw std::domain_error("Time offset exceeds audio duration");
  }

  // Calculate ending byte offset
  uint64_t end_byte;
  if (duration_seconds) {
    double dur_sec = *duration_seconds;
    if (dur_sec < 0.0) {
      throw std::domain_error("Duration cannot be negative");
    }

    uint64_t num_samples = static_cast<uint64_t>(dur_sec * header.sample_rate);
    uint64_t byte_length = num_samples * header.block_align;
    end_byte = start_byte + byte_length;
  } else {
    end_byte = available_data_size;
  }

  end_byte = std::min(end_byte, static_cast<uint64_t>(available_data_size));

  // Ensure we're aligned to block boundaries
  start_byte = (start_byte / header.block_align) * header.block_align;
  end_byte = (end_byte / header.block_align) * header.block_align;

  size_t length = static_cast<size_t>(end_byte - start_byte);
  return wav_data.substr(data_offset + start_byte, length);
}

} // namespace spdl::core
