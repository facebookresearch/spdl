#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace nb = nanobind;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// FOLLY INITIALIZATION
////////////////////////////////////////////////////////////////////////////////
struct DoublePtr {
  char **p, **p_orig;
  DoublePtr(int argc) : p(new char*[argc]), p_orig(p) {}
  DoublePtr(const DoublePtr&) = delete;
  DoublePtr& operator=(const DoublePtr&) = delete;
  DoublePtr(DoublePtr&&) noexcept = delete;
  DoublePtr& operator=(DoublePtr&&) noexcept = delete;
  ~DoublePtr() {
    delete[] p_orig;
  }
};

std::vector<std::string> init_folly_init(
    const std::string& prog,
    const std::vector<std::string>& orig_args) {
  int nargs = 1 + orig_args.size();
  DoublePtr args(nargs);
  args.p[0] = const_cast<char*>(prog.c_str());
  for (int i = 1; i < nargs; ++i) {
    args.p[i] = const_cast<char*>(orig_args[i - 1].c_str());
  }
  init_folly(&nargs, &args.p);

  std::vector<std::string> ret;
  for (int i = 0; i < nargs; ++i) {
    ret.emplace_back(args.p[i]);
  }
  return ret;
}
} // namespace

void register_utils(nb::module_& m) {
  m.def("init_folly", &init_folly_init);
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("register_avdevices", &register_avdevices);

  m.def("is_cuda_available", []() {
    nb::gil_scoped_release g;
    return is_cuda_available();
  });
  m.def("is_nvcodec_available", []() {
    nb::gil_scoped_release g;
    return is_nvcodec_available();
  });
}

} // namespace spdl::core
