#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;

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

void register_utils(py::module& m) {
  py::register_exception<spdl::core::InternalError>(
      m, "InternalError", PyExc_AssertionError);

  m.def("init_folly", &init_folly_init);
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("clear_ffmpeg_cuda_context_cache", &clear_ffmpeg_cuda_context_cache);
  m.def(
      "create_cuda_context",
      &create_cuda_context,
      py::arg("index"),
      py::arg("use_primary_context") = false);
  m.def("get_cuda_device_index", &get_cuda_device_index);
}

} // namespace spdl::core
