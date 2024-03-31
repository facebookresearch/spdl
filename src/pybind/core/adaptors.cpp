#include <libspdl/core/adaptor.h>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {

void register_adaptors(py::module& m) {
  auto _SourceAdaptor = py::class_<SourceAdaptor, SourceAdaptorPtr>(
      m, "SourceAdaptor", py::module_local());

  auto _MMapAdaptor =
      py::class_<MMapAdaptor, SourceAdaptor, std::shared_ptr<MMapAdaptor>>(
          m, "MMapAdaptor", py::module_local());

  auto _BytesAdaptor =
      py::class_<BytesAdaptor, SourceAdaptor, std::shared_ptr<BytesAdaptor>>(
          m, "BytesAdaptor", py::module_local());

  _SourceAdaptor.def(py::init<>());

  _MMapAdaptor.def(py::init<>());

  _BytesAdaptor.def(py::init<>());
}

} // namespace spdl::core
