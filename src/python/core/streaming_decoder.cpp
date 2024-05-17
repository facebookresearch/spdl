#include <libspdl/core/decoding.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::core {

void register_demuxer(nb::module_& m) {
  nb::class_<StreamingDecoder<MediaType::Video>>(m, "StreamingVideoDecoder")
      .def("decode", &StreamingDecoder<MediaType::Video>::decode);
}

} // namespace spdl::core
