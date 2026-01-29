/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_tar.h"
#include <fmt/core.h>
#include "tar_iterator.h"

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace spdl::archive {
namespace {

using TarEntry = nb::tuple;

struct FileObjTarParserImpl {
  nb::object fileobj_;
  size_t pos_ = 0;
  bool at_end_ = false;
  TarEntry entry_{};

 private:
  // Note on the GIL
  // - The GIL must be acquired when calling the `read` method, as it can
  //   do anything including allocating a Python object and reference counting.
  // - Additionally, the GIL also must be acquired when the returned bytes
  //   object is deleted, as Python will dereference its counter.
  // - Since the object destruction can happen at the end of the scope,
  //   this method must be called in the following way.
  //
  //   {
  //       nb::gil_scoped_acquire _;
  //       auto data = _read_exact(n);
  //   }
  //
  //   This way, the GIL is acquired while calling the `read` method, and
  //   while the resulting `data` object gets destroyed at the end of the scope.
  nb::bytes _read_exact(size_t n) {
    if (n == 0) {
      // zero-length return is reserved for EOF detection.
      // So we cannot pass zero to the `read` method.
      throw std::domain_error("[Internal Error] requested 0 read.");
    }
    auto data = nb::bytes(fileobj_.attr("read")(n));
    size_t num_read = data.size();
    pos_ += num_read;
    if (num_read > n) {
      throw std::runtime_error(
          fmt::format(
              "Received {} bytes which exceeds the requested size of {}.",
              num_read,
              n));
    }
    if (num_read < n) {
      throw std::runtime_error("Failed to fetch data.");
    }
    if (num_read == 0) {
      at_end_ = true;
    }
    return data;
  }

 public:
  void parse_next() {
    const TarHeader* header;
    std::string filepath;
    uint64_t file_size, padded_size;
    bool is_regular_file;
    std::string pending_long_name;

    while (!at_end_) {
      auto data = _read_exact(512);
      {
        nb::gil_scoped_release __;
        header = reinterpret_cast<const TarHeader*>(data.data());
        filepath = parse_filepath(header);
        if (filepath.size() == 0 && pending_long_name.empty()) {
          at_end_ = true;
          return;
        }
        file_size = parse_filesize(header);
        padded_size = (file_size + 511) & ~511ULL;
      }

      if (header->typeflag == 'L') {
        if (file_size > 0) {
          auto longname_data = _read_exact(file_size);
          {
            nb::gil_scoped_release __;
            pending_long_name =
                std::string(longname_data.c_str(), longname_data.size());
            if (!pending_long_name.empty() &&
                pending_long_name.back() == '\0') {
              pending_long_name.pop_back();
            }
          }
          if (auto remaining = padded_size - file_size; remaining) {
            _read_exact(remaining);
          }
        }
        continue;
      }

      if (header->typeflag == 'x') {
        if (file_size > 0) {
          auto pax_data = _read_exact(file_size);
          {
            nb::gil_scoped_release __;
            pending_long_name =
                parse_pax_path(pax_data.c_str(), pax_data.size());
          }
          if (auto remaining = padded_size - file_size; remaining) {
            _read_exact(remaining);
          }
        }
        continue;
      }

      is_regular_file = header->typeflag == '0' || header->typeflag == '\0';
      if (!is_regular_file) {
        pending_long_name.clear();
        if (padded_size) {
          _read_exact(padded_size);
        }
      } else {
        if (!pending_long_name.empty()) {
          filepath = std::move(pending_long_name);
          pending_long_name.clear();
        }
        entry_ = nb::make_tuple(
            nb::str(filepath.c_str(), filepath.size()),
            file_size ? _read_exact(file_size) : nb::bytes(""));
        if (auto remaining = padded_size - file_size; remaining) {
          _read_exact(remaining);
        }
        return;
      }
    }
  }

  const TarEntry& get_entry() const {
    return entry_;
  }

  bool done() const {
    return at_end_;
  }
};

class FileObjectTarParser {
  nb::object fileobj_;

 public:
  explicit FileObjectTarParser(nb::object& o) : fileobj_(o) {}

  TarParserIterator<FileObjTarParserImpl, const TarEntry&> begin() {
    return TarParserIterator<FileObjTarParserImpl, const TarEntry&>{
        FileObjTarParserImpl{fileobj_}};
  }

  static const TarEOF& end() {
    const static TarEOF placeholder;
    return placeholder;
  }
};
} // namespace

void register_tar(nb::module_& m) {
  m.doc() = "TAR file iterator for Python";

  nb::class_<InMemoryTarParser>(m, "InMemoryTarParser")
      .def(
          "__iter__",
          [](InMemoryTarParser& v) {
            auto begin = [&]() {
              nb::gil_scoped_release _;
              return v.begin();
            };

            auto end = [&]() {
              nb::gil_scoped_release _;
              return v.end();
            };

            return nb::make_iterator(
                nb::type<InMemoryTarParser>(), "iterator", begin(), end()
                //,nb::call_guard<nb::gil_scoped_release>()
            );
          },
          nb::keep_alive<0, 1>());

  m.def(
      "parse_tar",
      [](const nb::bytes& data) {
        return InMemoryTarParser{std::string_view{data.c_str(), data.size()}};
      }
      //,nb::call_guard<nb::gil_scoped_release>()
  );

  nb::class_<FileObjectTarParser>(m, "FileObjectTarParser")
      .def(
          "__iter__",
          [](FileObjectTarParser& v) {
            return nb::make_iterator(
                nb::type<FileObjectTarParser>(),
                "iterator",
                v.begin(),
                v.end());
          },
          nb::keep_alive<0, 1>());
  m.def("parse_tar", [](nb::object fileobj) {
    return FileObjectTarParser{fileobj};
  });
}

} // namespace spdl::archive
