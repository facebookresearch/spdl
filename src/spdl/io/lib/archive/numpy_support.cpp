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
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
  int major = static_cast<unsigned char>(d[0]);
  // int minor = static_cast<int>(data[1]);
  s -= 2;
  d += 2;
  switch (major) {
    case 1: {
      // The next two bytes are header length in little endien.
      if (s < 2) {
        throw std::runtime_error("Failed to parse header length.");
      }
      const size_t len = static_cast<unsigned char>(d[0]) |
          (static_cast<size_t>(static_cast<unsigned char>(d[1])) << 8);
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
      const size_t len = static_cast<unsigned char>(d[0]) |
          (static_cast<size_t>(static_cast<unsigned char>(d[1])) << 8) |
          (static_cast<size_t>(static_cast<unsigned char>(d[2])) << 16) |
          (static_cast<size_t>(static_cast<unsigned char>(d[3])) << 24);
      if (len == 0) {
        throw std::runtime_error(
            "Invalid data. The header length must be greater than 0.");
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

size_t skip_whitespace(const std::string_view value, size_t pos) {
  while (pos < value.size() &&
         std::isspace(static_cast<unsigned char>(value[pos]))) {
    ++pos;
  }
  return pos;
}

size_t parse_shape_dimension(std::string token) {
  const auto begin =
      std::find_if_not(token.begin(), token.end(), [](const unsigned char c) {
        return std::isspace(c);
      });
  if (begin == token.end()) {
    throw std::runtime_error("Failed to parse header `'shape'`.");
  }
  const auto end =
      std::find_if_not(token.rbegin(), token.rend(), [](unsigned char c) {
        return std::isspace(c);
      }).base();
  token = std::string(begin, end);
  if (!token.empty() && (token.back() == 'L' || token.back() == 'l')) {
    token.pop_back();
  }
  if (token.empty()) {
    throw std::runtime_error("Failed to parse header `'shape'`.");
  }
  const bool is_negative = token.front() == '-';
  if (token.front() == '+' || token.front() == '-') {
    token.erase(0, 1);
  }
  if (token.size() > 1 && token.front() == '0') {
    throw std::runtime_error("Failed to parse header `'shape'`.");
  }
  size_t dim{};
  const auto [parsed_end, error] =
      std::from_chars(token.data(), token.data() + token.size(), dim);
  if (token.empty() || error != std::errc{} ||
      parsed_end != token.data() + token.size() || (is_negative && dim != 0)) {
    throw std::runtime_error("Failed to parse header `'shape'`.");
  }
  return dim;
}

class HeaderParser {
  std::string_view header_;
  size_t pos_ = 0;
  NPYArray array_;
  bool has_descr_ = false;
  bool has_fortran_order_ = false;
  bool has_shape_ = false;

  [[noreturn]] static void fail() {
    throw std::runtime_error("Failed to parse NPY header dictionary.");
  }

  void skip_whitespace() {
    pos_ = spdl::archive::skip_whitespace(header_, pos_);
  }

  void expect(const char expected) {
    skip_whitespace();
    if (pos_ == header_.size() || header_[pos_] != expected) {
      fail();
    }
    ++pos_;
  }

  std::string_view parse_quoted_string() {
    skip_whitespace();
    if (pos_ == header_.size() ||
        (header_[pos_] != '\'' && header_[pos_] != '"')) {
      fail();
    }
    const char quote = header_[pos_++];
    const size_t begin = pos_;
    while (pos_ < header_.size() && header_[pos_] != quote) {
      if (header_[pos_] == '\\') {
        fail();
      }
      ++pos_;
    }
    if (pos_ == header_.size()) {
      fail();
    }
    const std::string_view value = header_.substr(begin, pos_ - begin);
    ++pos_;
    return value;
  }

  bool parse_boolean() {
    skip_whitespace();
    if (header_.substr(pos_, 4) == "True") {
      pos_ += 4;
      return true;
    }
    if (header_.substr(pos_, 5) == "False") {
      pos_ += 5;
      return false;
    }
    fail();
  }

  std::vector<size_t> parse_shape() {
    expect('(');
    skip_whitespace();
    if (pos_ < header_.size() && header_[pos_] == ')') {
      ++pos_;
      return {};
    }

    std::vector<size_t> shape;
    bool saw_comma = false;
    while (pos_ < header_.size()) {
      const size_t begin = pos_;
      while (pos_ < header_.size() && header_[pos_] != ',' &&
             header_[pos_] != ')') {
        ++pos_;
      }
      if (pos_ == header_.size()) {
        fail();
      }
      shape.push_back(parse_shape_dimension(
          std::string{header_.substr(begin, pos_ - begin)}));
      if (header_[pos_] == ')') {
        if (!saw_comma) {
          throw std::runtime_error(
              "Failed to parse header `'shape'` as a tuple.");
        }
        ++pos_;
        return shape;
      }

      saw_comma = true;
      ++pos_;
      skip_whitespace();
      if (pos_ < header_.size() && header_[pos_] == ')') {
        ++pos_;
        return shape;
      }
    }
    fail();
  }

  void parse_entry(const std::string_view key) {
    expect(':');
    if (key == "descr") {
      if (has_descr_) {
        fail();
      }
      skip_whitespace();
      if (pos_ < header_.size() && header_[pos_] == '[') {
        throw std::runtime_error("Structured NPY dtypes are not supported.");
      }
      array_.descr = parse_quoted_string();
      has_descr_ = true;
      return;
    }
    if (key == "fortran_order") {
      if (has_fortran_order_) {
        fail();
      }
      array_.fortran_order = parse_boolean();
      has_fortran_order_ = true;
      return;
    }
    if (key == "shape") {
      if (has_shape_) {
        fail();
      }
      array_.shape = parse_shape();
      has_shape_ = true;
      return;
    }
    throw std::runtime_error("Unexpected key in NPY header dictionary.");
  }

 public:
  explicit HeaderParser(const std::string_view header) : header_{header} {}

  NPYArray parse() {
    expect('{');
    while (true) {
      skip_whitespace();
      if (pos_ == header_.size() || header_[pos_] == '}') {
        fail();
      }
      parse_entry(parse_quoted_string());
      skip_whitespace();
      if (pos_ == header_.size()) {
        fail();
      }
      if (header_[pos_] == '}') {
        ++pos_;
        break;
      }
      if (header_[pos_] != ',') {
        fail();
      }
      ++pos_;
      skip_whitespace();
      if (pos_ < header_.size() && header_[pos_] == '}') {
        ++pos_;
        break;
      }
    }

    skip_whitespace();
    if (pos_ != header_.size() || !has_descr_ || !has_fortran_order_ ||
        !has_shape_) {
      fail();
    }
    return std::move(array_);
  }
};

NPYArray parse_header(const std::string_view header) {
  // NPY headers are Python dictionary literals with exactly these three keys.
  // Key order is intentionally unrestricted. See:
  // https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
  return HeaderParser{header}.parse();
}

std::optional<size_t> parse_decimal(
    const std::string_view value,
    const bool allow_zero) {
  if (value.empty()) {
    return std::nullopt;
  }
  size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      (!allow_zero && parsed == 0) ||
      parsed > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  return parsed;
}

bool is_divisor_of_any(
    const size_t denominator,
    const std::initializer_list<size_t> factors) {
  return std::any_of(
      factors.begin(), factors.end(), [denominator](const size_t factor) {
        return factor % denominator == 0;
      });
}

bool is_valid_datetime_denominator(
    const std::string_view base_unit,
    const size_t denominator) {
  if (denominator == 1) {
    return true;
  }
  if (base_unit == "Y") {
    return is_divisor_of_any(denominator, {12, 52, 365});
  }
  if (base_unit == "M") {
    return is_divisor_of_any(denominator, {4, 30, 720});
  }
  if (base_unit == "W") {
    return is_divisor_of_any(denominator, {7, 168, 10080});
  }
  if (base_unit == "D") {
    return is_divisor_of_any(denominator, {24, 1440, 86400});
  }
  if (base_unit == "h") {
    return is_divisor_of_any(denominator, {60, 3600});
  }
  if (base_unit == "m") {
    return is_divisor_of_any(denominator, {60, 60000});
  }
  if (base_unit == "s" || base_unit == "ms" || base_unit == "us" ||
      base_unit == "ns" || base_unit == "ps") {
    return is_divisor_of_any(denominator, {1000, 1000000});
  }
  if (base_unit == "fs") {
    return is_divisor_of_any(denominator, {1000});
  }
  return false;
}

bool is_valid_datetime_metadata_value(const std::string_view value) {
  size_t unit_begin = 0;
  while (unit_begin < value.size() &&
         std::isdigit(static_cast<unsigned char>(value[unit_begin]))) {
    ++unit_begin;
  }
  if (unit_begin != 0 &&
      !parse_decimal(value.substr(0, unit_begin), /*allow_zero=*/true)) {
    return false;
  }

  static constexpr std::string_view kDatetimeUnits[]{
      "Y",
      "M",
      "W",
      "D",
      "h",
      "m",
      "s",
      "ms",
      "us",
      "ns",
      "ps",
      "fs",
      "as",
      "generic"};
  const size_t slash_pos = value.find('/', unit_begin);
  const std::string_view base_unit = value.substr(
      unit_begin,
      slash_pos == std::string_view::npos ? slash_pos : slash_pos - unit_begin);
  if (std::find(
          std::begin(kDatetimeUnits), std::end(kDatetimeUnits), base_unit) ==
      std::end(kDatetimeUnits)) {
    return false;
  }
  if (unit_begin != 0 && base_unit == "generic") {
    return false;
  }
  if (slash_pos == std::string_view::npos) {
    return true;
  }
  const auto denominator =
      parse_decimal(value.substr(slash_pos + 1), /*allow_zero=*/false);
  return denominator && is_valid_datetime_denominator(base_unit, *denominator);
}

bool has_valid_datetime_metadata(
    const std::string_view descr,
    const size_t bracket_pos) {
  return bracket_pos < descr.size() && descr.size() - bracket_pos > 2 &&
      descr[bracket_pos] == '[' && descr.back() == ']' &&
      is_valid_datetime_metadata_value(
             descr.substr(bracket_pos + 1, descr.size() - bracket_pos - 2));
}

std::string_view strip_byte_order(const std::string_view descr) {
  return !descr.empty() &&
          std::string_view{"<>=|"}.find(descr.front()) != std::string_view::npos
      ? descr.substr(1)
      : descr;
}

bool is_datetime_alias(
    const std::string_view alias,
    const std::string_view name) {
  if (alias == name) {
    return true;
  }
  if (alias.size() <= name.size() + 2 || alias.substr(0, name.size()) != name ||
      alias[name.size()] != '[' || alias.back() != ']') {
    return false;
  }
  return is_valid_datetime_metadata_value(
      alias.substr(name.size() + 1, alias.size() - name.size() - 2));
}

std::optional<size_t> get_named_item_size(const std::string_view descr) {
  static constexpr std::pair<std::string_view, size_t> kNamedItemSizes[] = {
      {"?", 1},
      {"bool", 1},
      {"bool_", 1},
      {"byte", 1},
      {"ubyte", 1},
      {"int8", 1},
      {"uint8", 1},
      {"short", sizeof(short)},
      {"ushort", sizeof(unsigned short)},
      {"int16", 2},
      {"uint16", 2},
      {"half", 2},
      {"float16", 2},
      {"int32", 4},
      {"intc", sizeof(int)},
      {"uint32", 4},
      {"uintc", sizeof(unsigned int)},
      {"single", 4},
      {"float32", 4},
      {"int", sizeof(intptr_t)},
      {"int_", sizeof(intptr_t)},
      {"intp", sizeof(intptr_t)},
      {"int0", sizeof(intptr_t)},
      {"uint", sizeof(uintptr_t)},
      {"uintp", sizeof(uintptr_t)},
      {"uint0", sizeof(uintptr_t)},
      {"long", sizeof(long)},
      {"ulong", sizeof(unsigned long)},
      {"longlong", sizeof(long long)},
      {"ulonglong", sizeof(unsigned long long)},
      {"int64", 8},
      {"uint64", 8},
      {"float", 8},
      {"double", 8},
      {"float64", 8},
      {"float_", 8},
      {"longdouble", sizeof(long double)},
      {"longfloat", sizeof(long double)},
      {"csingle", 8},
      {"singlecomplex", 8},
      {"complex64", 8},
      {"complex", 16},
      {"cdouble", 16},
      {"complex128", 16},
      {"complex_", 16},
      {"clongdouble", 2 * sizeof(long double)},
      {"clongfloat", 2 * sizeof(long double)},
      {"longcomplex", 2 * sizeof(long double)},
      {"datetime64", 8},
      {"timedelta64", 8},
      {"bytes", 0},
      {"bytes_", 0},
      {"str", 0},
      {"str_", 0},
      {"unicode", 0},
      {"unicode_", 0},
      {"void", 0},
      {"void0", 0},
  };
  for (const auto& [name, item_size] : kNamedItemSizes) {
    if (descr == name) {
      return item_size;
    }
  }
  if ((descr == "float128" && sizeof(long double) == 16) ||
      (descr == "float96" && sizeof(long double) == 12)) {
    return sizeof(long double);
  }
  if ((descr == "complex256" && sizeof(long double) == 16) ||
      (descr == "complex192" && sizeof(long double) == 12)) {
    return 2 * sizeof(long double);
  }

  const std::string_view alias = strip_byte_order(descr);
  if (is_datetime_alias(alias, "datetime64") ||
      is_datetime_alias(alias, "timedelta64")) {
    return 8;
  }
  return std::nullopt;
}

std::optional<size_t> get_typecode_item_size(const std::string_view descr) {
  const bool has_byte_order = !descr.empty() &&
      (descr[0] == '<' || descr[0] == '>' || descr[0] == '=' ||
       descr[0] == '|');
  const std::string_view typecode = descr.substr(has_byte_order ? 1 : 0);
  if (typecode.size() != 1) {
    return std::nullopt;
  }

  switch (typecode[0]) {
    case '?':
    case 'b':
    case 'B':
    case 'c':
      return 1;
    case 'h':
    case 'H':
      return sizeof(short);
    case 'i':
    case 'I':
      return sizeof(int);
    case 'l':
    case 'L':
      return sizeof(long);
    case 'q':
    case 'Q':
      return sizeof(long long);
    case 'p':
    case 'P':
    case 'n':
    case 'N':
      return sizeof(intptr_t);
    case 'e':
      return 2;
    case 'f':
      return 4;
    case 'd':
      return 8;
    case 'g':
      return sizeof(long double);
    case 'F':
      return 8;
    case 'D':
      return 16;
    case 'G':
      return 2 * sizeof(long double);
    case 'm':
    case 'M':
      return 8;
    case 'a':
    case 'S':
    case 'U':
    case 'V':
      return 0;
    case 'O':
      throw std::runtime_error("Object NPY dtypes are not supported.");
    default:
      return std::nullopt;
  }
}

std::pair<char, size_t> parse_dtype_kind(const std::string_view descr) {
  const bool has_byte_order = !descr.empty() &&
      (descr[0] == '<' || descr[0] == '>' || descr[0] == '=' ||
       descr[0] == '|');
  const size_t kind_pos = has_byte_order ? 1 : 0;
  if (kind_pos == descr.size()) {
    throw std::runtime_error("Unsupported NPY dtype descriptor.");
  }
  const char kind = descr[kind_pos];
  if (kind == 'O') {
    throw std::runtime_error("Object NPY dtypes are not supported.");
  }
  if (std::string_view{"biufcmMSUaV"}.find(kind) == std::string_view::npos) {
    throw std::runtime_error("Unsupported NPY dtype descriptor.");
  }
  return {kind, kind_pos};
}

bool contains_item_size(
    const size_t item_size,
    const std::initializer_list<size_t> supported_sizes) {
  return std::find(supported_sizes.begin(), supported_sizes.end(), item_size) !=
      supported_sizes.end();
}

bool is_supported_item_size(const char kind, const size_t item_size) {
  switch (kind) {
    case 'b':
      return item_size == 1;
    case 'i':
    case 'u':
      return contains_item_size(item_size, {1, 2, 4, 8});
    case 'f':
      return contains_item_size(item_size, {2, 4, 8, sizeof(long double)});
    case 'c':
      return contains_item_size(item_size, {8, 16, 2 * sizeof(long double)});
    case 'm':
    case 'M':
      return item_size == 8;
    case 'S':
    case 'U':
    case 'a':
    case 'V':
      return true;
  }
  return false;
}

bool permits_zero_item_size(const char kind) {
  return std::string_view{"SUaV"}.find(kind) != std::string_view::npos;
}

size_t parse_item_size_digits(
    const std::string_view descr,
    const char kind,
    const size_t kind_pos) {
  const size_t digit_begin = kind_pos + 1;
  size_t digit_end = digit_begin;
  while (digit_end < descr.size() &&
         std::isdigit(static_cast<unsigned char>(descr[digit_end]))) {
    ++digit_end;
  }
  const bool has_datetime_unit = (kind == 'm' || kind == 'M') &&
      has_valid_datetime_metadata(descr, digit_end);
  if (digit_end == digit_begin ||
      (digit_end != descr.size() && !has_datetime_unit)) {
    throw std::runtime_error("Unsupported NPY dtype descriptor.");
  }

  size_t item_size{};
  const auto [end, error] = std::from_chars(
      descr.data() + digit_begin, descr.data() + digit_end, item_size);
  if (error != std::errc{} || end != descr.data() + digit_end ||
      (item_size == 0 && !permits_zero_item_size(kind))) {
    throw std::runtime_error("Unsupported NPY dtype descriptor.");
  }

  if (!is_supported_item_size(kind, item_size)) {
    throw std::runtime_error("Unsupported NPY dtype descriptor.");
  }
  return item_size;
}

std::pair<char, size_t> parse_numeric_item_size(const std::string_view descr) {
  const auto [kind, kind_pos] = parse_dtype_kind(descr);
  return {kind, parse_item_size_digits(descr, kind, kind_pos)};
}

size_t get_item_size(const std::string_view descr) {
  if (const auto item_size = get_named_item_size(descr)) {
    return *item_size;
  }
  if (const auto item_size = get_typecode_item_size(descr)) {
    return *item_size;
  }
  auto [kind, item_size] = parse_numeric_item_size(descr);
  if (kind == 'U') {
    if (item_size > std::numeric_limits<size_t>::max() / 4) {
      throw std::runtime_error("NPY dtype size exceeds the supported range.");
    }
    item_size *= 4;
  }
  return item_size;
}

size_t get_data_size(const NPYArray& array) {
  size_t num_items = 1;
  for (const size_t dim : array.shape) {
    if (dim != 0 && num_items > std::numeric_limits<size_t>::max() / dim) {
      throw std::runtime_error("NPY array shape exceeds the supported range.");
    }
    num_items *= dim;
  }

  // NumPy permits zero-width flexible dtypes such as |V0. Shape overflow is
  // still validated above even when the resulting payload is empty.
  if (array.item_size != 0 &&
      num_items > std::numeric_limits<size_t>::max() / array.item_size) {
    throw std::runtime_error("NPY array size exceeds the supported range.");
  }
  return num_items * array.item_size;
}
} // namespace

NPYArray load_npy(const char* data, size_t size) {
  check_magic(&data, &size);
  auto header = extract_header(&data, &size);
  auto array = parse_header(header);
  array.item_size = get_item_size(array.descr);
  if (size < get_data_size(array)) {
    throw std::runtime_error("NPY payload is shorter than its declared shape.");
  }
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
