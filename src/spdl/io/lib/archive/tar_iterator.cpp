/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tar_iterator.h"

#include <cstddef>
#include <cstring>

namespace spdl::archive {
//////////////////////////////////////////////////////////////////////////////
// TAR format and helper functions
//////////////////////////////////////////////////////////////////////////////
namespace {
uint32_t calculate_checksum(const TarHeader* header) {
  uint32_t checksum = 0;
  const char* bytes = reinterpret_cast<const char*>(header);

  for (size_t i = 0; i < sizeof(TarHeader); ++i) {
    if (i >= offsetof(TarHeader, checksum) &&
        i < offsetof(TarHeader, checksum) + sizeof(header->checksum)) {
      checksum += ' ';
    } else {
      checksum += static_cast<unsigned char>(bytes[i]);
    }
  }

  return checksum;
}

uint64_t parse_octal(const char* str, size_t len) {
  uint64_t result = 0;
  for (size_t i = 0; i < len && str[i] != '\0' && str[i] != ' '; ++i) {
    if (str[i] >= '0' && str[i] <= '7') {
      result = result * 8 + (str[i] - '0');
    }
  }
  return result;
}

} // namespace

bool is_valid_header(const TarHeader* header) {
  if (std::strncmp(header->magic, "ustar", 5) != 0 &&
      std::strncmp(header->magic, "ustar ", 6) != 0) {
    return false;
  }

  uint64_t stored_checksum =
      parse_octal(header->checksum, sizeof(header->checksum));
  uint64_t calculated_checksum = calculate_checksum(header);

  return stored_checksum == calculated_checksum;
}
std::string parse_filepath(const TarHeader* header) {
  if (header->name[0] == '\0') {
    return {};
  }

  std::string filename;
  if (header->prefix[0] != '\0') {
    filename = std::string(
        header->prefix, strnlen(header->prefix, sizeof(header->prefix)));
    if (filename.back() != '/') {
      filename += '/';
    }
  }
  filename +=
      std::string(header->name, strnlen(header->name, sizeof(header->name)));
  return filename;
}

uint64_t parse_filesize(const TarHeader* header) {
  return parse_octal(header->size, sizeof(header->size));
}

std::string parse_pax_path(const char* data, size_t size) {
  std::string_view sv(data, size);
  const std::string_view path_key = "path=";

  size_t pos = 0;
  while (pos < size) {
    size_t space_pos = sv.find(' ', pos);
    if (space_pos == std::string_view::npos) {
      break;
    }

    size_t record_start = space_pos + 1;
    if (sv.substr(record_start, path_key.size()) == path_key) {
      size_t value_start = record_start + path_key.size();
      size_t newline_pos = sv.find('\n', value_start);
      if (newline_pos != std::string_view::npos) {
        return std::string(sv.substr(value_start, newline_pos - value_start));
      }
    }

    size_t newline = sv.find('\n', pos);
    if (newline == std::string_view::npos) {
      break;
    }
    pos = newline + 1;
  }
  return {};
}

//////////////////////////////////////////////////////////////////////////////
// Core implementation
//////////////////////////////////////////////////////////////////////////////

InMemoryTarParserImpl::InMemoryTarParserImpl(const std::string_view& s)
    : data_(s.data()), size_(s.size()) {}

bool InMemoryTarParserImpl::done() const {
  return at_end_;
}

const TarIndex& InMemoryTarParserImpl::get_entry() const {
  return current_entry_;
}

void InMemoryTarParserImpl::parse_next() {
  if (at_end_) {
    return;
  }

  std::string pending_long_name;

  while (pos_ + sizeof(TarHeader) <= size_) {
    const TarHeader* header = reinterpret_cast<const TarHeader*>(data_ + pos_);

    if (!is_valid_header(header)) {
      pos_ += 512;
      continue;
    }

    auto filepath = parse_filepath(header);
    if (filepath.size() == 0 && pending_long_name.empty()) {
      at_end_ = true;
      return;
    }

    uint64_t file_size = parse_filesize(header);
    uint64_t padded_size = (file_size + 511) & ~511ULL;

    if (header->typeflag == 'L') {
      pos_ += 512;
      if (pos_ + file_size > size_) {
        at_end_ = true;
        return;
      }
      pending_long_name = std::string(data_ + pos_, file_size);
      if (!pending_long_name.empty() && pending_long_name.back() == '\0') {
        pending_long_name.pop_back();
      }
      pos_ += padded_size;
      continue;
    }

    if (header->typeflag == 'x') {
      pos_ += 512;
      if (pos_ + file_size > size_) {
        at_end_ = true;
        return;
      }
      pending_long_name = parse_pax_path(data_ + pos_, file_size);
      pos_ += padded_size;
      continue;
    }

    if (header->typeflag == '0' || header->typeflag == '\0') {
      pos_ += 512;

      if (!pending_long_name.empty()) {
        filepath = std::move(pending_long_name);
        pending_long_name.clear();
      }

      if (pos_ + file_size > size_) {
        at_end_ = true;
        return;
      }

      current_entry_ = std::make_tuple(std::move(filepath), pos_, file_size);

      pos_ += padded_size;
      return;
    } else {
      pending_long_name.clear();
      pos_ += 512 + padded_size;
    }
  }

  at_end_ = true;
}

//////////////////////////////////////////////////////////////////////////////
// Iterable interface
//////////////////////////////////////////////////////////////////////////////
InMemoryTarParser::InMemoryTarParser(const std::string_view& s) : data_(s) {}

TarParserIterator<InMemoryTarParserImpl, const TarIndex&>
InMemoryTarParser::begin() const {
  return TarParserIterator<InMemoryTarParserImpl, const TarIndex&>(
      InMemoryTarParserImpl{data_});
}

const TarEOF& InMemoryTarParser::end() {
  const static TarEOF placeholder;
  return placeholder;
}

} // namespace spdl::archive
