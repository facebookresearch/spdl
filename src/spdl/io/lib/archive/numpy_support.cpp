/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "numpy_support.h"
#include "zip_impl.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace spdl::archive {

//////////////////////////////////////////////////////////////////////////////
// load_npy
//////////////////////////////////////////////////////////////////////////////

namespace {

void check_magic(const char** data, size_t* size) {
  const static char* prefix = "\x93NUMPY";
  const static size_t len = std::strlen(prefix);
  if (*size < len) {
    throw std::runtime_error(
        "Failed to parse the magic prefix. (data too short)");
  }
  if (std::strncmp(*data, prefix, len) != 0) {
    throw std::runtime_error(
        "The data must start with the prefix '\\x93NUMPY'");
  }
  *data = (*data) + len;
  *size = (*size) - len;
}

std::string_view extract_header(const char** data, size_t* size) {
  auto s = (*size);
  auto* d = (*data);
  if (s < 2) {
    throw std::runtime_error("Failed to parse version number.");
  }
  int major = static_cast<int>(d[0]);
  // int minor = static_cast<int>(data[1]);
  s -= 2;
  d += 2;
  switch (major) {
    case 1: {
      // The next two bytes are header length in little endien.
      if (s < 2) {
        throw std::runtime_error("Failed to parse header length.");
      }
      unsigned short len = (*d);
      len += (unsigned short)(*(d + 1)) << 8;
      s -= 2;
      d += 2;
      if (s < len) {
        throw std::runtime_error("Failed to parse header");
      }
      std::string_view header{d, len};
      *data = d + len;
      *size = s - len;
      return header;
    }
    case 2:
      [[fallthrough]];
    case 3: {
      // The next four bytes are header length.
      if (s < 4) {
        throw std::runtime_error("Failed to parse header length.");
      }
      size_t len;
      {
        int l = (int)*d;
        l += (int)(*(d + 1) << 8);
        l += (int)(*(d + 2) << 16);
        l += (int)(*(d + 3) << 24);
        if (l <= 0) {
          throw std::runtime_error(
              "Invalid data. The header length must be greater than 0.");
        }
        len = l;
      }
      s -= 4;
      d += 4;
      if (s < len) {
        throw std::runtime_error("Failed to parse header");
      }
      std::string_view header{d, len};
      *data = d + len;
      *size = s - len;
      return header;
    }
    default:
      throw std::runtime_error(
          "Unexpected format version. Only 1, 2 and 3 are supported.");
  }
}

NPYArray parse_header(const std::string_view header) {
  // NPY header is a string expression of Python dictionary with the following
  // keys See:
  // https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
  //
  // - "descr": Format description. e.g. "'<i8'", "'<f4'"
  // - "fortran_order": "True" or "False".
  // - "shape": Tuple of int. e.g.  "()", "(3, 4, 5)"
  NPYArray ret;
  {
    size_t pos = header.find("'descr':");
    if (pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'descr'`.");
    }
    pos = header.find('\'', pos + 7);
    if (pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'descr'`.");
    }
    size_t end_pos = header.find('\'', pos + 1);
    if (end_pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'descr'`.");
    }
    ret.descr = header.substr(pos + 1, end_pos - pos - 1);
  }
  {
    size_t pos = header.find("'shape':");
    if (pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'shape'`.");
    }
    pos = header.find('(', pos);
    if (pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'shape'`.");
    }
    size_t end_pos = header.find(')', pos);
    if (end_pos == std::string::npos) {
      throw std::runtime_error("Failed to parse header `'shape'`.");
    }
    std::string shape_str(header.substr(pos + 1, end_pos - pos - 1));
    std::istringstream shape_stream(shape_str);
    std::string number;
    while (std::getline(shape_stream, number, ',')) {
      number.erase(
          std::remove_if(number.begin(), number.end(), ::isspace),
          number.end());
      if (!number.empty()) {
        ret.shape.push_back(std::stoi(number));
      }
    }
  }
  {
    const std::string key = "'fortran_order':";
    size_t pos = header.find(key);
    if (pos != std::string::npos) {
      pos += key.length();
      while (pos < header.size() && std::isspace(header[pos])) {
        ++pos;
      }
      if (pos < header.size() && header[pos] == 'T') {
        ret.fortran_order = true;
      } else if (pos < header.size() && header[pos] == 'F') {
        ret.fortran_order = false;
      }
    }
  }
  return ret;
}
} // namespace

NPYArray load_npy(const char* data, size_t size) {
  check_magic(&data, &size);
  auto header = extract_header(&data, &size);
  auto array = parse_header(header);
  array.data = (void*)data;
  return array;
}

NPYArray load_npy_compressed(
    const char* data,
    uint32_t compressed_size,
    uint32_t uncompressed_size) {
  auto buffer = std::make_unique<char[]>(uncompressed_size);
  zip::inflate(data, compressed_size, buffer.get(), uncompressed_size);
  auto ret = load_npy(buffer.get(), uncompressed_size);
  ret.buffer = std::move(buffer);
  return ret;
}

} // namespace spdl::archive
