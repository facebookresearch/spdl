/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "zip_impl.h"

extern "C" {
#include <libdeflate.h> // @manual
#include <zlib.h>
}

#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace spdl::archive::zip {
namespace {

constexpr uint32_t kCdSignature = 0x02014b50; // PK\x01\x02
constexpr uint32_t kLfhSignature = 0x04034b50; // PK\x03\x04
constexpr uint32_t kEocdSignature = 0x06054b50; // PK\x05\x06
constexpr size_t kMinEocdSize = 22;
constexpr size_t kMaxCommentSize = 65535;

bool contains_region(
    const size_t size,
    const size_t offset,
    const size_t length) {
  return offset <= size && length <= size - offset;
}

uint16_t read_u16(const char* data, const size_t size, const size_t offset) {
  if (data == nullptr) {
    throw std::runtime_error("Invalid ZIP data: null buffer.");
  }
  if (!contains_region(size, offset, 2)) {
    throw std::runtime_error("Invalid ZIP data: truncated 16-bit field.");
  }
  const auto* p = reinterpret_cast<const unsigned char*>(data + offset);
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32(const char* data, const size_t size, const size_t offset) {
  if (data == nullptr) {
    throw std::runtime_error("Invalid ZIP data: null buffer.");
  }
  if (!contains_region(size, offset, 4)) {
    throw std::runtime_error("Invalid ZIP data: truncated 32-bit field.");
  }
  const auto* p = reinterpret_cast<const unsigned char*>(data + offset);
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
      (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const char* data, const size_t size, const size_t offset) {
  if (data == nullptr) {
    throw std::runtime_error("Invalid ZIP data: null buffer.");
  }
  if (!contains_region(size, offset, 8)) {
    throw std::runtime_error("Invalid ZIP data: truncated 64-bit field.");
  }
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(static_cast<unsigned char>(data[offset + i]))
        << (8 * i);
  }
  return value;
}

// Note: The structure of End of Central Directory
// Offset Bytes Description
//    0    4    End of central directory signature = 0x06054b50
//    4    2    Number of this disk (or 0xffff for ZIP64)
//    6    2    Disk where central directory starts (or 0xffff for ZIP64)
//    8    2    Number of central directory records on this disk
//              (or 0xffff for ZIP64)
//   10    2    Total number of central directory records
//              (or 0xffff for ZIP64)
//   12    4    Size of central directory (bytes) (or 0xffffffff for ZIP64)
//   16    4    Offset of start of central directory,
//              relative to start of archive (or 0xffffffff for ZIP64)
//   20    2    Comment length (n)
//   22    n    Comment

struct EOCD {
  size_t cd_size;
  size_t cd_offset;
  size_t num_entries;
  size_t offset;
  uint16_t disk;
  uint16_t cd_disk;
  size_t disk_entries;
};

enum class EocdKind { supported, multi_disk, zip64 };

EocdKind classify_eocd(const EOCD& eocd) {
  if (eocd.num_entries == std::numeric_limits<uint16_t>::max() ||
      eocd.cd_size == std::numeric_limits<uint32_t>::max() ||
      eocd.cd_offset == std::numeric_limits<uint32_t>::max()) {
    return EocdKind::zip64;
  }
  if (eocd.disk != 0 || eocd.cd_disk != 0 ||
      eocd.disk_entries != eocd.num_entries) {
    return EocdKind::multi_disk;
  }
  return EocdKind::supported;
}

void validate_eocd(const EOCD& eocd) {
  switch (classify_eocd(eocd)) {
    case EocdKind::supported:
      return;
    case EocdKind::multi_disk:
      throw std::runtime_error("Multi-disk ZIP archives are not supported.");
    case EocdKind::zip64:
      throw std::runtime_error("ZIP64 central directories are not supported.");
  }
}

bool central_directory_ends_at_eocd(const EOCD& eocd) {
  return eocd.cd_offset <= eocd.offset &&
      eocd.cd_size == eocd.offset - eocd.cd_offset;
}

std::optional<EOCD>
try_parse_eocd_at(const char* root, const size_t len, const size_t offset) {
  if (read_u32(root, len, offset) != kEocdSignature) {
    return std::nullopt;
  }
  const size_t comment_size = read_u16(root, len, offset + 20);
  if (comment_size > len - offset - kMinEocdSize) {
    return std::nullopt;
  }

  return EOCD{
      .cd_size = read_u32(root, len, offset + 12),
      .cd_offset = read_u32(root, len, offset + 16),
      .num_entries = read_u16(root, len, offset + 10),
      .offset = offset,
      .disk = read_u16(root, len, offset + 4),
      .cd_disk = read_u16(root, len, offset + 6),
      .disk_entries = read_u16(root, len, offset + 8)};
}

EOCD parse_eocd(const char* root, size_t len) {
  if (len < kMinEocdSize) {
    throw std::runtime_error("The data is not a valid zip file.");
  }

  const size_t last_offset = len - kMinEocdSize;
  const size_t first_offset = len > kMinEocdSize + kMaxCommentSize
      ? len - kMinEocdSize - kMaxCommentSize
      : 0;
  std::optional<EOCD> supported_candidate;
  std::optional<EOCD> fallback_candidate;
  for (size_t offset = last_offset;; --offset) {
    if (const auto eocd = try_parse_eocd_at(root, len, offset)) {
      fallback_candidate = eocd;
      if (classify_eocd(*eocd) == EocdKind::supported &&
          central_directory_ends_at_eocd(*eocd)) {
        supported_candidate = eocd;
      }
    }
    if (offset == first_offset) {
      break;
    }
  }
  if (supported_candidate) {
    return *supported_candidate;
  }
  if (fallback_candidate) {
    validate_eocd(*fallback_candidate);
    throw std::runtime_error(
        "Invalid ZIP data: central directory does not end at the EOCD.");
  }
  throw std::runtime_error(
      "Failed to locate the end of the central directory.");
}

// The structure of Central directory file header
//
// Offset  Bytes  Description
//      0    4    Central directory file header signature = 0x02014b50
//      4    2    Version made by
//      6    2    Version needed to extract (minimum)
//      8    2    General purpose bit flag
//     10    2    Compression method
//     12    2    File last modification time
//     14    2    File last modification date
//     16    4    CRC-32 of uncompressed data
//     20    4    Compressed size (or 0xffffffff for ZIP64)
//     24    4    Uncompressed size (or 0xffffffff for ZIP64)
//     28    2    File name length (n)
//     30    2    Extra field length (m)
//     32    2    File comment length (k)
//     34    2    Disk number where file starts (or 0xffff for ZIP64)
//     36    2    Internal file attributes
//     38    4    External file attributes
//     42    4    Relative offset of local file header
//                (or 0xffffffff for ZIP64).
//                This is the number of bytes between the start of
//                the first disk on which the file occurs,
//                and the start of the local file header.
//                This allows software reading the central directory
//                to locate the position of the file inside the ZIP file.
//     46    n    File name
//   46+n    m    Extra field
// 46+n+m    k    File comment

struct CDFH {
  uint64_t local_header_offset;
  uint64_t compressed_size;
  uint64_t uncompressed_size;
  uint32_t crc32;
  uint16_t compression_method;
  uint16_t filename_length;
  uint16_t disk;
  size_t extra_field_offset;
  size_t extra_field_length;
  size_t size;
};

CDFH parse_cdh(
    const char* root,
    const size_t len,
    const size_t offset,
    const size_t limit) {
  if (!contains_region(limit, offset, 46)) {
    throw std::out_of_range(
        "Invalid data found. "
        "The central directory extends to the outside of specified region.");
  }
  if (read_u32(root, len, offset) != kCdSignature) {
    throw std::runtime_error(
        "Failed to locate the central directory. (Signature does not match).");
  }
  const uint16_t filename_length = read_u16(root, len, offset + 28);
  const uint16_t extra_field_length = read_u16(root, len, offset + 30);
  const uint16_t comment_length = read_u16(root, len, offset + 32);
  const size_t size = 46 + static_cast<size_t>(filename_length) +
      extra_field_length + comment_length;
  if (!contains_region(limit, offset, size)) {
    throw std::domain_error(
        "Invalid data found. "
        "The central directory record extends to the outside of the given data.");
  }
  // Central-directory size and offset fields are 32-bit on disk. ZIP64 stores
  // 0xffffffff sentinels here; parse_zip64_extended_info resolves them to the
  // corresponding 64-bit values before they are used.
  return CDFH{
      .local_header_offset = read_u32(root, len, offset + 42),
      .compressed_size = read_u32(root, len, offset + 20),
      .uncompressed_size = read_u32(root, len, offset + 24),
      .crc32 = read_u32(root, len, offset + 16),
      .compression_method = read_u16(root, len, offset + 10),
      .filename_length = filename_length,
      .disk = read_u16(root, len, offset + 34),
      .extra_field_offset = offset + 46 + filename_length,
      .extra_field_length = extra_field_length,
      .size = size};
}

// The structure of Local file header
// Offset  Bytes Description
//    0    4     Local file header signature = 0x04034b50
//    4    2     Version needed to extract (minimum)
//    6    2     General purpose bit flag
//    8    2     Compression method;
//               e.g. none = 0, DEFLATE = 8 (or "\0x08\0x00")
//   10    2     File last modification time
//   12    2     File last modification date
//   14    4     CRC-32 of uncompressed data
//   18    4     Compressed size (or 0xffffffff for ZIP64)
//   22    4     Uncompressed size (or 0xffffffff for ZIP64)
//   26    2     File name length (n)
//   28    2     Extra field length (m)
//   30    n     File name
// 30+n    m     Extra field

struct LOC {
  uint16_t compression_method;
  size_t size;
};

LOC parse_loc(const char* root, const size_t len, const uint64_t raw_offset) {
  if (raw_offset > len) {
    throw std::domain_error(
        "Invalid data found. The local file header is outside the archive.");
  }
  const size_t offset = static_cast<size_t>(raw_offset);
  if (!contains_region(len, offset, 30)) {
    throw std::domain_error(
        "Invalid data found. The local file header is truncated.");
  }
  if (read_u32(root, len, offset) != kLfhSignature) {
    throw std::domain_error(
        "Failed to locate the local file header. (Signature does not match).");
  }
  const size_t size = 30 +
      static_cast<size_t>(read_u16(root, len, offset + 26)) +
      read_u16(root, len, offset + 28);
  if (!contains_region(len, offset, size)) {
    throw std::domain_error(
        "Invalid data found. The local file header is truncated.");
  }
  return LOC{
      .compression_method = read_u16(root, len, offset + 8), .size = size};
}

// Offset  Bytes  Description[37]
//      0      2  Header ID 0x0001
//      2      2  Size of the extra field chunk (8, 16, 24 or 28)
//      4      8  Original uncompressed file size
//     12      8  Size of compressed data
//     20      8  Offset of local header record
//     28      4  Number of the disk on which this file starts
struct Zip64Meta {
  uint64_t uncompressed_size;
  uint64_t compressed_size;
  uint64_t local_header_offset;
  uint32_t disk;
};

Zip64Meta parse_zip64_extended_info(
    const char* root,
    const size_t len,
    const CDFH& cdfh) {
  Zip64Meta result{
      .uncompressed_size = cdfh.uncompressed_size,
      .compressed_size = cdfh.compressed_size,
      .local_header_offset = cdfh.local_header_offset,
      .disk = cdfh.disk};
  const bool needs_uncompressed =
      cdfh.uncompressed_size == std::numeric_limits<uint32_t>::max();
  const bool needs_compressed =
      cdfh.compressed_size == std::numeric_limits<uint32_t>::max();
  const bool needs_offset =
      cdfh.local_header_offset == std::numeric_limits<uint32_t>::max();
  const bool needs_disk = cdfh.disk == std::numeric_limits<uint16_t>::max();
  if (!needs_uncompressed && !needs_compressed && !needs_offset &&
      !needs_disk) {
    return result;
  }

  const size_t extra_limit = cdfh.extra_field_offset + cdfh.extra_field_length;
  size_t offset = cdfh.extra_field_offset;
  while (offset < extra_limit) {
    if (!contains_region(extra_limit, offset, 4)) {
      throw std::domain_error("Invalid ZIP extra field.");
    }
    const uint16_t tag = read_u16(root, len, offset);
    const size_t field_size = read_u16(root, len, offset + 2);
    const size_t value_offset = offset + 4;
    if (!contains_region(extra_limit, value_offset, field_size)) {
      throw std::domain_error("Invalid ZIP extra field.");
    }
    if (tag == 0x0001) {
      const size_t expected_size = (needs_uncompressed ? 8 : 0) +
          (needs_compressed ? 8 : 0) + (needs_offset ? 8 : 0) +
          (needs_disk ? 4 : 0);
      if (field_size != expected_size) {
        throw std::domain_error("Invalid ZIP64 extended metadata.");
      }
      size_t cursor = value_offset;
      const size_t field_limit = value_offset + field_size;
      // The ZIP64 values appear in APPNOTE order only when the corresponding
      // central-directory field uses its maximum-value sentinel. Every such
      // required value must fit entirely inside this extra field.
      const auto require_remaining = [&](const size_t width) {
        if (!contains_region(field_limit, cursor, width)) {
          throw std::domain_error("Invalid ZIP64 extended metadata.");
        }
      };
      auto take_u64 = [&]() {
        require_remaining(8);
        const uint64_t value = read_u64(root, field_limit, cursor);
        cursor += 8;
        return value;
      };
      auto take_u32 = [&]() {
        require_remaining(4);
        const uint32_t value = read_u32(root, field_limit, cursor);
        cursor += 4;
        return value;
      };
      if (needs_uncompressed) {
        result.uncompressed_size = take_u64();
      }
      if (needs_compressed) {
        result.compressed_size = take_u64();
      }
      if (needs_offset) {
        result.local_header_offset = take_u64();
      }
      if (needs_disk) {
        result.disk = take_u32();
      }
      return result;
    }
    offset = value_offset + field_size;
  }
  throw std::domain_error("Failed to locate the ZIP64 extended metadata.");
}
} // namespace

std::vector<ZipMetaData> parse_zip(const char* root, const size_t len) {
  if (root == nullptr) {
    throw std::runtime_error("Invalid ZIP data: null buffer.");
  }
  const auto eocd = parse_eocd(root, len);
  size_t cd_offset = eocd.cd_offset;
  if (!contains_region(eocd.offset, cd_offset, eocd.cd_size)) {
    throw std::domain_error(
        "Invalid data found. "
        "The central directory extends to the outside of the given data.");
  }
  const size_t cd_limit = cd_offset + eocd.cd_size;

  std::vector<ZipMetaData> ret;
  ret.reserve(eocd.num_entries);
  for (size_t i = 0; i < eocd.num_entries; ++i) {
    const auto cdfh = parse_cdh(root, len, cd_offset, cd_limit);
    std::string_view filename{root + cd_offset + 46, cdfh.filename_length};

    const auto metadata = parse_zip64_extended_info(root, len, cdfh);
    if (metadata.disk != 0) {
      throw std::runtime_error("Multi-disk ZIP archives are not supported.");
    }
    const auto loc = parse_loc(root, len, metadata.local_header_offset);
    if (loc.compression_method != cdfh.compression_method) {
      throw std::domain_error(
          "The local and central ZIP compression methods do not match.");
    }
    const size_t local_header_offset =
        static_cast<size_t>(metadata.local_header_offset);
    const size_t file_start = local_header_offset + loc.size;
    if (file_start > eocd.cd_offset ||
        metadata.compressed_size > eocd.cd_offset - file_start) {
      throw std::domain_error(
          "Invalid data found. The file payload is outside the archive.");
    }
    ret.emplace_back(
        filename,
        file_start,
        metadata.compressed_size,
        metadata.uncompressed_size,
        cdfh.compression_method,
        cdfh.crc32);
    cd_offset += cdfh.size;
  }
  return ret;
}

void verify_crc32(const char* data, size_t size, uint32_t expected) {
  if (libdeflate_crc32(0, data, size) != expected) {
    throw std::runtime_error("ZIP entry failed CRC-32 validation.");
  }
}

namespace {

struct Decompressor {
  libdeflate_decompressor* p;

  Decompressor() : p(libdeflate_alloc_decompressor()) {
    if (!p) {
      throw std::runtime_error("Failed to allocate decompressor.");
    }
  }
  Decompressor(const Decompressor&) = default;
  Decompressor(Decompressor&&) = delete;
  Decompressor& operator=(const Decompressor&) = default;
  Decompressor& operator=(Decompressor&&) = delete;

  ~Decompressor() {
    libdeflate_free_decompressor(p);
  }
};

} // namespace

void inflate(
    const char* src,
    size_t compressed_size,
    void* dst,
    size_t uncompressed_size) {
  Decompressor d{};
  size_t actual_decompressed_size = 0;
  enum libdeflate_result result = libdeflate_deflate_decompress(
      d.p,
      src,
      compressed_size,
      dst,
      uncompressed_size,
      &actual_decompressed_size);

  if (result != LIBDEFLATE_SUCCESS ||
      actual_decompressed_size != uncompressed_size) {
    throw std::runtime_error("Failed to decompress the data");
  }
}

} // namespace spdl::archive::zip
