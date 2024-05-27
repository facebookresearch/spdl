#include <libspdl/core/cuda.h>
#include <libspdl/core/encoding.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using u8_cpu_array = nb::ndarray<uint8_t, nb::device::cpu, nb::c_contig>;
using u8_cuda_array = nb::ndarray<uint8_t, nb::device::cuda, nb::c_contig>;

namespace spdl::core {
namespace {

template <typename... Ts>
std::vector<size_t> get_shape(nb::ndarray<Ts...>& arr) {
  std::vector<size_t> ret;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    ret.push_back(arr.shape(i));
  }
  return ret;
}

void encode(
    std::string path,
    u8_cpu_array data,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg) {
  auto src = reinterpret_cast<void*>(data.data());
  auto shape = get_shape(data);
  nb::gil_scoped_release g;
  encode_image(path, src, shape, pix_fmt, encode_cfg);
}

void encode_cuda(
    std::string uri,
    u8_cuda_array data,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg) {
  auto src = reinterpret_cast<void*>(data.data());
  auto shape = get_shape(data);
  nb::gil_scoped_release g;
  auto storage = cp_to_cpu(src, shape);
  encode_image(uri, storage.data(), shape, pix_fmt, encode_cfg);
}
} // namespace

void register_encoding(nb::module_& m) {
  m.def(
      "encode_image",
      &encode,
      nb::arg("path"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none());

  m.def(
      "encode_image",
      &encode_cuda,
      nb::arg("path"),
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none());
}

} // namespace spdl::core
