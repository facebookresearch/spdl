/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::archive::zip {

using ZipMetaData = std::tuple<
    std::string, // filename
    uint64_t, // offset
    uint64_t, // compressed_size
    uint64_t, // uncompressed_size
    uint16_t, // compression_method
    uint32_t // CRC-32
    >;

std::vector<ZipMetaData> parse_zip(const char* root, const size_t len);

void verify_crc32(const char* data, size_t size, uint32_t expected);

void inflate(
    const char* root,
    size_t compressed_size,
    void* dst,
    size_t uncompressed_size);

} // namespace spdl::archive::zip
