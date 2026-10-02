/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_utils.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace spdl::core {

namespace {

template <typename T>
T read_little_endian(const char* data) {
  static_assert(std::is_unsigned_v<T>);
  T value{};
  for (size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<T>(static_cast<unsigned char>(data[i])) << (8 * i);
  }
  return value;
}

size_t get_next_chunk_offset(
    std::string_view wav_data,
    size_t offset,
    const char* out_of_bounds_message) {
  const uint32_t chunk_size =
      read_little_endian<uint32_t>(wav_data.data() + offset + 4);
  const size_t data_offset = offset + 8;
  if (chunk_size > wav_data.size() - data_offset) {
    throw std::domain_error(out_of_bounds_message);
  }

  const size_t data_end = data_offset + chunk_size;
  const size_t padding = chunk_size & 1U;
  if (padding > wav_data.size() - data_end) {
    throw std::domain_error(out_of_bounds_message);
  }
  return data_end + padding;
}

bool check_fourcc(std::string_view data, size_t offset, const char* expected) {
  if (offset + 4 > data.size()) {
    return false;
  }
  return std::memcmp(data.data() + offset, expected, 4) == 0;
}

void validate_wav_header_for_loading(const WAVHeader& header) {
  if (header.num_channels == 0) {
    throw std::domain_error("Invalid number of channels: 0");
  }
  if (header.sample_rate == 0) {
    throw std::domain_error("Invalid sample rate: 0");
  }

  switch (header.audio_format) {
    case 1: // PCM
      if (header.bits_per_sample != 8 && header.bits_per_sample != 16 &&
          header.bits_per_sample != 32) {
        throw std::domain_error(
            "Unsupported PCM bits per sample: " +
            std::to_string(header.bits_per_sample));
      }
      break;
    case 3: // IEEE float
      if (header.bits_per_sample != 32 && header.bits_per_sample != 64) {
        throw std::domain_error(
            "Unsupported IEEE float bits per sample: " +
            std::to_string(header.bits_per_sample));
      }
      break;
    default:
      throw std::domain_error(
          "Unsupported WAV audio format: " +
          std::to_string(header.audio_format));
  }

  const uint64_t expected_block_align =
      static_cast<uint64_t>(header.num_channels) * (header.bits_per_sample / 8);
  if (header.block_align != expected_block_align) {
    throw std::domain_error(
        "Invalid block align: expected " +
        std::to_string(expected_block_align) + ", found " +
        std::to_string(header.block_align));
  }

  const uint64_t expected_byte_rate =
      static_cast<uint64_t>(header.sample_rate) * expected_block_align;
  if (header.byte_rate != expected_byte_rate) {
    throw std::domain_error(
        "Invalid byte rate: expected " + std::to_string(expected_byte_rate) +
        ", found " + std::to_string(header.byte_rate));
  }
}

long double sample_position(double seconds, uint32_t sample_rate) {
  return static_cast<long double>(seconds) * sample_rate;
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

  while (offset <= wav_data.size() && wav_data.size() - offset >= 8) {
    if (check_fourcc(wav_data, offset, "fmt ")) {
      const uint32_t fmt_size =
          read_little_endian<uint32_t>(wav_data.data() + offset + 4);
      const size_t next_offset = get_next_chunk_offset(
          wav_data, offset, "fmt chunk extends beyond file size");

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
      offset = next_offset;
      break;
    }
    offset = get_next_chunk_offset(
        wav_data, offset, "WAV chunk extends beyond file size");
  }

  if (!found_fmt) {
    throw std::domain_error("fmt chunk not found");
  }

  // Find data chunk
  while (offset <= wav_data.size() && wav_data.size() - offset >= 8) {
    if (check_fourcc(wav_data, offset, "data")) {
      header.data_size =
          read_little_endian<uint32_t>(wav_data.data() + offset + 4);
      const size_t data_offset = offset + 8;
      if (header.data_size > wav_data.size() - data_offset) {
        throw std::domain_error(
            "WAV data chunk extends beyond the input buffer");
      }
      return header;
    }

    offset = get_next_chunk_offset(
        wav_data, offset, "WAV chunk extends beyond file size");
  }

  throw std::domain_error("data chunk not found");
}

std::string_view extract_wav_samples(
    std::string_view wav_data,
    std::optional<double> time_offset_seconds,
    std::optional<double> duration_seconds) {
  // Parse the WAV header
  WAVHeader header = parse_wav_header(wav_data);

  validate_wav_header_for_loading(header);

  // Find the start of the data chunk
  size_t offset = 12;
  size_t data_offset = 0;

  while (offset <= wav_data.size() && wav_data.size() - offset >= 8) {
    if (check_fourcc(wav_data, offset, "data")) {
      data_offset = offset + 8;
      break;
    }

    offset = get_next_chunk_offset(
        wav_data, offset, "WAV chunk extends beyond the input buffer");
  }

  if (data_offset == 0) {
    throw std::domain_error("data chunk not found");
  }

  const size_t bytes_after_header = wav_data.size() - data_offset;
  if (header.data_size > bytes_after_header) {
    throw std::domain_error("WAV data chunk extends beyond the input buffer");
  }
  const size_t available_data_size = header.data_size;
  if (available_data_size % header.block_align != 0) {
    throw std::domain_error(
        "WAV data chunk does not contain a whole number of sample frames");
  }

  // If no time window specified, return entire waveform
  if (!time_offset_seconds && !duration_seconds) {
    return wav_data.substr(data_offset, available_data_size);
  }

  // Calculate byte offsets for time window
  const double offset_sec = time_offset_seconds.value_or(0.0);
  if (!std::isfinite(offset_sec)) {
    throw std::domain_error("Time offset must be finite");
  }
  if (offset_sec < 0.0) {
    throw std::domain_error("Time offset cannot be negative");
  }

  const size_t available_samples = available_data_size / header.block_align;
  const long double raw_start_sample =
      sample_position(offset_sec, header.sample_rate);
  if (raw_start_sample >= available_samples) {
    throw std::domain_error("Time offset exceeds audio duration");
  }
  const size_t start_sample = static_cast<size_t>(raw_start_sample);

  size_t end_sample = available_samples;
  if (duration_seconds) {
    const double dur_sec = *duration_seconds;
    if (!std::isfinite(dur_sec)) {
      throw std::domain_error("Duration must be finite");
    }
    if (dur_sec < 0.0) {
      throw std::domain_error("Duration cannot be negative");
    }

    const long double raw_num_samples =
        sample_position(dur_sec, header.sample_rate);
    const size_t remaining_samples = available_samples - start_sample;
    if (raw_num_samples < remaining_samples) {
      end_sample = start_sample + static_cast<size_t>(raw_num_samples);
    }
  }

  const size_t start_byte = start_sample * header.block_align;
  const size_t length = (end_sample - start_sample) * header.block_align;
  return wav_data.substr(data_offset + start_byte, length);
}

} // namespace spdl::core
