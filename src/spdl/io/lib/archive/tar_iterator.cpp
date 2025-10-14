/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tar_iterator.h"
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

  while (pos_ + sizeof(TarHeader) <= size_) {
    const TarHeader* header = reinterpret_cast<const TarHeader*>(data_ + pos_);

    if (!is_valid_header(header)) {
      pos_ += 512;
      continue;
    }

    auto filepath = parse_filepath(header);
    if (filepath.size() == 0) {
      at_end_ = true;
      return;
    }

    uint64_t file_size = parse_filesize(header);

    if (header->typeflag == '0' || header->typeflag == '\0') {
      pos_ += 512;

      if (pos_ + file_size > size_) {
        at_end_ = true;
        return;
      }

      current_entry_ = std::make_tuple(std::move(filepath), pos_, file_size);

      uint64_t padded_size = (file_size + 511) & ~511ULL;
      pos_ += padded_size;
      return;
    } else {
      pos_ += 512;
      uint64_t padded_size = (file_size + 511) & ~511ULL;
      pos_ += padded_size;
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
