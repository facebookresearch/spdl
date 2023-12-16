#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <libspdl/processors.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl {
namespace {

py::buffer_info get_buffer(VideoBuffer& b) {
  if (b.channel_last) {
    return {
        b.data.data(),
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        4,
        {b.n, b.h, b.w, b.c},
        {sizeof(uint8_t) * b.c * b.w * b.h,
         sizeof(uint8_t) * b.c * b.w,
         sizeof(uint8_t) * b.c,
         sizeof(uint8_t)}};
  }
  return {
      b.data.data(),
      sizeof(uint8_t),
      py::format_descriptor<uint8_t>::format(),
      4,
      {b.n, b.c, b.h, b.w},
      {sizeof(uint8_t) * b.w * b.h * b.c,
       sizeof(uint8_t) * b.w * b.h,
       sizeof(uint8_t) * b.w,
       sizeof(uint8_t)}};
}

struct DoublePtr {
  char** p;
  DoublePtr(int argc) : p(new char*[argc]) {}
  DoublePtr(const DoublePtr&) = delete;
  DoublePtr& operator=(const DoublePtr&) = delete;
  DoublePtr(DoublePtr&&) noexcept = delete;
  DoublePtr& operator=(DoublePtr&&) noexcept = delete;
  ~DoublePtr() {
    delete[] p;
  }
};

folly::Init* FOLLY_INIT = nullptr;

void delete_folly_init() {
  delete FOLLY_INIT;
}
  
void init_folly_init(const std::string& prog, const std::vector<std::string>& orig_args) {
  int nargs = 1 + orig_args.size();
  DoublePtr args(nargs);
  args.p[0] = const_cast<char*>(prog.c_str());
  for (size_t i = 1; i < nargs; ++i) {
    args.p[i] = const_cast<char*>(orig_args[i-1].c_str());
  }
  FOLLY_INIT = new folly::Init{&nargs, &args.p, false};
  Py_AtExit(delete_folly_init);
}

PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  m.def(
      "init_folly",
      [](const std::string& prog,
         const std::optional<std::vector<std::string>>& args){
        init_folly_init(prog, args.value_or(std::vector<std::string>{}));
      },
      py::arg("prog"),
      py::arg("args") = py::none());
  py::class_<VideoBuffer>(
      m, "VideoBuffer", py::buffer_protocol(), py::module_local())
      .def_buffer(get_buffer);
  py::class_<Engine>(m, "Engine", py::module_local())
      .def(
          py::init([](size_t num_io_threads,
                      size_t num_decoding_threads,
                      size_t frame_queue_size) {
            return new Engine{
                num_io_threads, num_decoding_threads, frame_queue_size};
          }),
          py::arg("num_io_threads"),
          py::arg("num_decoding_threads"),
          py::arg("frame_queue_size"))
      .def(
          "enqueue",
          [](Engine& self,
             const std::string& src,
             const std::vector<double>& timestamps,
             const std::optional<std::string>& format = std::nullopt,
             const std::optional<OptionDict>& format_options = std::nullopt,
             const int buffer_size = 8096,
             const std::optional<std::string>& decoder = std::nullopt,
             const std::optional<OptionDict>& decoder_options = std::nullopt,
             const int cuda_device_index = -1,
             const std::optional<std::tuple<int, int>>& frame_rate =
                 std::nullopt,
             const std::optional<int>& width = std::nullopt,
             const std::optional<int>& height = std::nullopt,
             const std::optional<std::string>& pix_fmt = std::nullopt) {
            self.enqueue(
                {src,
                 timestamps,
                 format,
                 format_options,
                 buffer_size,
                 decoder,
                 decoder_options,
                 cuda_device_index,
                 frame_rate,
                 width,
                 height,
                 pix_fmt});
          },
          py::arg("src"),
          py::arg("timestamps"),
          py::arg("format") = py::none(),
          py::arg("format_options") = py::none(),
          py::arg("buffer_size") = 8096,
          py::arg("decoder") = py::none(),
          py::arg("decoder_options") = py::none(),
          py::arg("cuda_device_index") = -1,
          py::arg("frame_rate") = py::none(),
          py::arg("width") = py::none(),
          py::arg("height") = py::none(),
          py::arg("pix_fmt") = py::none())
      .def("dequeue", &Engine::dequeue);
}
} // namespace
} // namespace spdl
