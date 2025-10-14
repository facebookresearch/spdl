/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <tuple>

namespace spdl::archive {

//////////////////////////////////////////////////////////////////////////////
// TAR format and helper functions
//////////////////////////////////////////////////////////////////////////////

struct alignas(1) TarHeader {
  char name[100];
  char mode[8];
  char uid[8];
  char gid[8];
  char size[12];
  char mtime[12];
  char checksum[8];
  char typeflag;
  char linkname[100];
  char magic[6];
  char version[2];
  char uname[32];
  char gname[32];
  char devmajor[8];
  char devminor[8];
  char prefix[155];
  char padding[12];
};

static_assert(sizeof(TarHeader) == 512, "TAR header must be 512 bytes");

bool is_valid_header(const TarHeader* header);
std::string parse_filepath(const TarHeader*);
uint64_t parse_filesize(const TarHeader*);

//////////////////////////////////////////////////////////////////////////////
// Core implementation
//////////////////////////////////////////////////////////////////////////////

using TarIndex = std::tuple<std::string, size_t, size_t>;

struct InMemoryTarParserImpl {
  void parse_next();
  const TarIndex& get_entry() const;
  bool done() const;

  explicit InMemoryTarParserImpl(const std::string_view& tar_data_);

 private:
  const char* data_;
  size_t size_;
  size_t pos_ = 0;
  TarIndex current_entry_ = {};
  bool at_end_ = false;
};

//////////////////////////////////////////////////////////////////////////////
// Iterator interface
//////////////////////////////////////////////////////////////////////////////

struct TarEOF {}; // Just a placeholder. EOF marker is in parser itself.

template <typename ParserImpl, typename V>
class TarParserIterator {
 public:
  explicit TarParserIterator(ParserImpl&& p) : parser_(std::move(p)) {
    parser_.parse_next();
  }

  TarParserIterator& operator++() {
    parser_.parse_next();
    return *this;
  }

  V operator*() const {
    return parser_.get_entry();
  }

  bool operator==(const TarEOF&) const {
    return parser_.done();
  }

 private:
  ParserImpl parser_;
};

//////////////////////////////////////////////////////////////////////////////
// Iterable interface
//////////////////////////////////////////////////////////////////////////////
class InMemoryTarParser {
  const std::string_view data_;

 public:
  explicit InMemoryTarParser(const std::string_view& s);

  TarParserIterator<InMemoryTarParserImpl, const TarIndex&> begin() const;
  static const TarEOF& end();
};

} // namespace spdl::archive
