/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

extern "C" {
#include <zlib.h>
}

#define CD_SIGNATURE 0x02014b50 // PK\x01\x02
#define LFH_SIGNATURE 0x04034b50 // PK\x03\x04
#define EOCD_SIGNATURE 0x06054b50 // PK\x05\x06

namespace spdl::zip {

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
  uint32_t cd_size;
  uint32_t cd_offset;
  uint16_t num_entries;
};

EOCD parse_eocd(const char* root, size_t len) {
#define MAX_COMMENT_LEN 65535
#define MIN_EOCD_LEN 22
  if (len < MIN_EOCD_LEN) {
    throw std::runtime_error("The data is not a valid zip file.");
  }
  // Iterate from the back until the EOCD signature is found.
  for (size_t i = 0; i <= len - MIN_EOCD_LEN && i < MAX_COMMENT_LEN; i += 4) {
    auto* eocd = root + len - MIN_EOCD_LEN - i;
    if (EOCD_SIGNATURE == *((uint32_t*)(eocd))) {
      return EOCD{
          .cd_size = *((uint32_t*)(eocd + 12)),
          .cd_offset = *((uint32_t*)(eocd + 16)),
          .num_entries = *((uint16_t*)(eocd + 10))};
    }
  }
  throw std::runtime_error(
      "Failed to locate the end of the central directory.");
#undef MAX_COMMENT_LEN
#undef MIN_EOCD_LEN
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
  uint32_t local_header_offset;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t compression_method;
  uint16_t filename_length;
  uint16_t size;
};

CDFH parse_cdh(const char* cd) {
  if (CD_SIGNATURE != *((uint32_t*)cd)) {
    throw std::runtime_error(
        "Failed to locate the central directory. (Signature does not match).");
  }
  auto filename_length = *((uint16_t*)(cd + 28));
  auto extra_field_length = *((uint16_t*)(cd + 30));
  auto comment_length = *((uint16_t*)(cd + 32));
  uint16_t size = 46 + filename_length + extra_field_length + comment_length;
  return CDFH{
      .local_header_offset = *((uint32_t*)(cd + 42)),
      .compressed_size = *((uint32_t*)(cd + 20)),
      .uncompressed_size = *((uint32_t*)(cd + 24)),
      .compression_method = *((uint16_t*)(cd + 10)),
      .filename_length = filename_length,
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
  // uint32_t compressed_size;
  // uint32_t uncompressed_size;
  // uint16_t flag;
  // uint16_t filename_length;
  // uint16_t extra_field_length;
  uint16_t size;
};

LOC parse_loc(const char* p) {
  if (LFH_SIGNATURE != *((uint32_t*)p)) {
    throw std::domain_error(
        "Failed to locate the local file header. (Signature does not match).");
  }
  uint16_t filename_length = *((uint16_t*)(p + 26));
  uint16_t extra_field_length = *((uint16_t*)(p + 28));
  uint16_t size = 30 + filename_length + extra_field_length;
  return LOC{
      .size = size
      // .compressed_size = *((uint32_t*)(p + 18)),
      // .uncompressed_size = *((uint32_t*)(p + 22)),
      // .flag = *((uint16_t*)(p + 6)),
      // .filename_length = *((uint16_t*)(p + 26)),
      // .extra_field_length = *((uint16_t*)(p + 28)),
  };
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
};

Zip64Meta parse_zip64_extended_info(const char* p) {
  if (0x0001 != *((uint16_t*)(p))) {
    throw std::domain_error(
        "Failed to locate the Zip 64 extended metadata. (Signature does not match).");
  }
  if (auto size = *((uint16_t*)(p + 2)); size < 16) {
    throw std::domain_error(
        "Invalid data found. Failed to fetch uncompressed data size.");
  }
  return Zip64Meta{
      .uncompressed_size = *((uint64_t*)(p + 4)),
      .compressed_size = *((uint64_t*)(p + 12))};
}

using ZipMetaData = std::tuple<
    std::string, // filename
    uint64_t, // offset
    uint64_t, // compressed_size
    uint64_t, // uncompressed_size
    uint16_t // compression_method
    >;

std::vector<ZipMetaData> parse_zip(const char* root, const size_t len) {
  auto eocd = parse_eocd(root, len);
  auto cd_offset = eocd.cd_offset;
  auto cd_limit = eocd.cd_offset + eocd.cd_size;
  if (cd_limit > len) {
    throw std::domain_error(
        "Invalid data found. "
        "The central directory extends to the outside of the given data.");
  }

  std::vector<ZipMetaData> ret;
  for (uint16_t i = 0; i < eocd.num_entries; ++i) {
    if (cd_offset + 46 > cd_limit) {
      throw std::out_of_range(
          "Invalid data found. "
          "The central directory extends to the outside of specified region.");
    }
    auto cdfh = parse_cdh(root + cd_offset);
    if (cdfh.local_header_offset + 30 > len) {
      throw std::domain_error(
          "Invalid data found. "
          "The central directory record points to a data region outside of the given data.");
    }
    if (cd_offset + 46 + cdfh.filename_length > len) {
      throw std::domain_error(
          "Invalid data found. "
          "The central directory record extends to the outside of the given data.");
    }
    std::string_view filename{root + cd_offset + 46, cdfh.filename_length};

    auto [compressed_size, uncompressed_size] =
        [&]() -> std::tuple<uint64_t, uint64_t> {
      if (cdfh.compressed_size == 0xFFFFFFFF) {
        auto info = parse_zip64_extended_info(root + cd_offset + cdfh.size);
        return {info.compressed_size, info.uncompressed_size};
      }
      return {cdfh.compressed_size, cdfh.uncompressed_size};
    }();

    auto loc = parse_loc(root + cdfh.local_header_offset);

    auto file_start = cdfh.local_header_offset + loc.size;
    ret.emplace_back(ZipMetaData{
        filename,
        file_start,
        compressed_size,
        uncompressed_size,
        cdfh.compression_method});
    cd_offset += cdfh.size;
  }
  return ret;
}

namespace {
int inflate(
    const void* src,
    uint32_t compressed_size,
    void* dst,
    uint32_t uncompressed_size,
    int windowBits) {
  z_stream strm;
  strm.total_in = strm.avail_in = compressed_size;
  strm.total_out = strm.avail_out = uncompressed_size;
  strm.next_in = (Bytef*)src;
  strm.next_out = (Bytef*)dst;

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  if (int err = inflateInit2(&strm, windowBits); err != Z_OK) {
    inflateEnd(&strm);
    return err;
  }

  if (int err = inflate(&strm, Z_FINISH); err == Z_STREAM_END) {
    inflateEnd(&strm);
    return Z_OK;
  } else {
    inflateEnd(&strm);
    return err;
  }
}

} // namespace

void inflate(
    const char* src,
    uint32_t compressed_size,
    void* dst,
    uint32_t uncompressed_size) {
  auto ret = inflate(src, compressed_size, dst, uncompressed_size, -15);
  switch (ret) {
    case Z_MEM_ERROR:
      throw std::runtime_error(
          "Failed to inflate the data due to out of memory.");
    case Z_BUF_ERROR:
      throw std::runtime_error(
          "Failed to inflate the data. There is not enough room in the output buffer.");
    case Z_DATA_ERROR:
      throw std::domain_error(
          "Failed to inflate the data. the input data is corrupted or incomplete.");
  }
}

} // namespace spdl::zip
