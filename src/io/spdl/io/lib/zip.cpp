/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/core.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <memory>
#include <vector>

#include <glog/logging.h>

#if __has_include(<experimental/source_location>)
#include <experimental/source_location>
using std::experimental::source_location;
#elif defined __cpp_lib_source_location
#include <source_location>
using std::source_location;
#else
#include <experimental/source_location>
using std::experimental::source_location;
#endif

extern "C" {
#include <zip.h>
}

namespace nb = nanobind;

namespace {

class ZipError {
  zip_error_t err;

 public:
  ZipError() {
    zip_error_init(&err);
  }
  ~ZipError() {
    zip_error_fini(&err);
  }
  ZipError(const ZipError&) = delete;
  ZipError& operator=(const ZipError&) = delete;
  ZipError(ZipError&& other) noexcept = delete;
  ZipError& operator=(ZipError&& other) noexcept = delete;
  operator zip_error_t*() {
    return &err;
  }
};

struct ZipInfo {
  zip_stat_t s;
  ZipInfo() {
    zip_stat_init(&s);
  }
  operator zip_stat_t*() {
    return &s;
  }
};

class ZipFile {
  zip_file_t* p;

 public:
  explicit ZipFile(zip_file_t* p) : p(p) {}
  explicit ZipFile(const ZipFile&) = delete;
  ZipFile& operator=(const ZipFile&) = delete;
  ZipFile(ZipFile&& other) noexcept = delete;
  ZipFile& operator=(ZipFile&& other) noexcept = delete;
  ~ZipFile() {
    close();
  }
  int64_t read(void* buf, uint64_t size) {
    int64_t num_read = zip_fread(p, buf, size);
    if (num_read < 0) {
      throw std::runtime_error(fmt::format("Failed to read file."));
    }
    return num_read;
  }

  int close() {
    int ret = 0;
    if (p) {
      ret = zip_fclose(p);
      if (ret == 0) {
        p = nullptr;
      }
    }
    return ret;
  }
};

class ZipArchive {
  zip_t* archive;

 public:
  explicit ZipArchive(zip_t* archive) : archive(archive) {}
  ZipArchive(const ZipArchive&) = delete;
  ZipArchive& operator=(const ZipArchive&) = delete;
  ZipArchive(ZipArchive&& other) noexcept = delete;
  ZipArchive& operator=(ZipArchive&& other) noexcept = delete;
  ~ZipArchive() {
    if (zip_close(archive) < 0) {
      LOG(WARNING) << fmt::format(
          "Failed to close archive: {}", zip_strerror(archive));
    }
  }

  std::vector<std::string> namelist() {
    std::vector<std::string> names;
    for (int i = 0; i < zip_get_num_entries(archive, 0); ++i) {
      names.push_back(zip_get_name(archive, i, ZIP_FL_ENC_GUESS));
    }
    return names;
  }

  std::shared_ptr<ZipFile> open(const std::string& name) {
    auto* fp = zip_fopen(archive, name.c_str(), 0);
    if (!fp) {
      throw std::runtime_error(
          fmt::format("Failed to open file: {}", zip_strerror(archive)));
    }
    return std::make_shared<ZipFile>(fp);
  }

  ZipInfo getinfo(const std::string& name) {
    ZipInfo info;
    if (zip_stat(archive, name.c_str(), 0, info) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to stat file: {}", zip_strerror(archive)));
    }
    return info;
  }
};

std::shared_ptr<ZipArchive> zip_archive(void* data, size_t size) {
  ZipError error;
  auto src = zip_source_buffer_create(data, size, 0, error);
  if (!src) {
    throw std::runtime_error(
        fmt::format("Failed to create source: {}", zip_error_strerror(error)));
  }

  auto* archive = zip_open_from_source(src, ZIP_RDONLY, error);
  if (!archive) {
    zip_source_free(src);
    throw std::runtime_error(
        fmt::format("Failed to open archive: {}", zip_error_strerror(error)));
  }
  return std::make_shared<ZipArchive>(archive);
}

NB_MODULE(_zip, m) {
  nb::class_<ZipInfo>(m, "ZipInfo")
      .def_prop_ro(
          "size",
          [](ZipInfo& self) {
            nb::gil_scoped_release __g;

            return (self.s.valid | ZIP_STAT_SIZE) ? self.s.size : 0;
          })
      .def_prop_ro(
          "comp_size",
          [](ZipInfo& self) {
            nb::gil_scoped_release __g;

            return (self.s.valid | ZIP_STAT_COMP_SIZE) ? self.s.comp_size : 0;
          })
      .def_prop_ro("comp_method", [](ZipInfo& self) -> std::string {
        nb::gil_scoped_release __g;

        if (self.s.valid | ZIP_STAT_COMP_METHOD) {
          switch (self.s.comp_method) {
            case ZIP_CM_DEFAULT:
              return "DEFAULT";
            case ZIP_CM_STORE:
              return "STORE";
            case ZIP_CM_BZIP2:
              return "BZIP2";
            case ZIP_CM_DEFLATE:
              return "DEFLATE";
            case ZIP_CM_XZ:
              return "XZ";
              // case ZIP_CM_ZSTD:
              //   return "ZSTD";
          }
        }
        return "UNKNOWN";
      });
  nb::class_<ZipFile>(m, "ZipFile")
      .def(
          "read",
          [](ZipFile& self, nb::bytes buffer) {
            auto* data = (void*)buffer.c_str();
            auto size = buffer.size();

            nb::gil_scoped_release __g;
            return self.read(data, size);
          })
      .def("close", [](ZipFile& self) {
        nb::gil_scoped_release __g;
        return self.close();
      });
  nb::class_<ZipArchive>(m, "ZipArchive")
      .def(
          "namelist",
          [](ZipArchive& self) {
            nb::gil_scoped_release __g;
            return self.namelist();
          })
      .def(
          "open",
          [](ZipArchive& self, const std::string& name) {
            nb::gil_scoped_release __g;
            return self.open(name);
          })
      .def("getinfo", [](ZipArchive& self, const std::string& name) {
        nb::gil_scoped_release __g;
        return self.getinfo(name);
      });

  m.def("zip_archive", [](nb::bytes bytes) {
    auto* data = (void*)bytes.c_str();
    auto size = bytes.size();

    nb::gil_scoped_release __g;
    return zip_archive(data, size);
  });
}

} // namespace
