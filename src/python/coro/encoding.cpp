#include <libspdl/coro/encoding.h>

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

namespace spdl::coro {
namespace {

template <typename... Ts>
std::vector<size_t> shape(nb::ndarray<Ts...>& arr) {
  std::vector<size_t> ret;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    ret.push_back(arr.shape(i));
  }
  return ret;
}

} // namespace

void register_encoding(nb::module_& m) {
  m.def(
      "async_encode_image",
      [](std::function<void(int)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::string uri,
         u8_cpu_array data,
         const std::string& pix_fmt,
         const std::optional<EncodeConfig>& encode_cfg,
         ThreadPoolExecutorPtr executor) -> FuturePtr {
        return async_encode_image(
            std::move(set_result),
            std::move(notify_exception),
            uri,
            reinterpret_cast<void*>(data.data()),
            shape(data),
            pix_fmt,
            encode_cfg,
            executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("uri"),
      nb::arg("data"),
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none(),
      nb::arg("executor") = nullptr);

  m.def(
      "async_encode_image",
      [](std::function<void(int)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         std::string uri,
         u8_cuda_array data,
         const std::string& pix_fmt,
         const std::optional<EncodeConfig>& encode_cfg,
         ThreadPoolExecutorPtr executor) -> FuturePtr {
        return async_encode_image_cuda(
            std::move(set_result),
            std::move(notify_exception),
            uri,
            reinterpret_cast<void*>(data.data()),
            shape(data),
            pix_fmt,
            encode_cfg,
            executor);
      },
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("uri"),
      nb::arg("data"),
      nb::arg("pix_fmt") = "rgb24",
      nb::arg("encode_config") = nb::none(),
      nb::arg("executor") = nullptr);
}

} // namespace spdl::coro
