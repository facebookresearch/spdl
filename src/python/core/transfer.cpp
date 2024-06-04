#include <libspdl/core/transfer.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {
CUDABufferPtr _transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg) {
  nb::gil_scoped_release g;
  return transfer_buffer(std::move(buffer), cfg);
}
} // namespace

void register_transfer(nb::module_& m) {
  m.def(
      "transfer_buffer",
      &_transfer_buffer,
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("cuda_config"));
}
} // namespace spdl::core
