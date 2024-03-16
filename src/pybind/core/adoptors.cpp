#include <libspdl/core/adoptor/basic.h>
#include <libspdl/core/adoptor/bytes.h>
#include <libspdl/core/adoptor/mmap.h>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Trampoline class for registering abstract SourceAdoptor
////////////////////////////////////////////////////////////////////////////////
class PySourceAdoptor : public SourceAdoptor {
 public:
  using SourceAdoptor::SourceAdoptor;

  void* get(const std::string& url, const IOConfig& io_cfg) const override {
    PYBIND11_OVERLOAD_PURE(void*, SourceAdoptor, get, url, io_cfg);
  }
};

} // namespace

void register_adoptors(py::module& m) {
  // SourceAdoptor is used by external libraries to provide customized source.
  // This registeration is global.
  // To reduce the possibilty of name colision, suffixing with `_SPDL_GLOBAL`.
  auto _SourceAdoptor =
      py::class_<SourceAdoptor, PySourceAdoptor, SourceAdoptorPtr>(
          m, "SourceAdoptor_SPDL_GLOBAL");

  auto _BasicAdoptor =
      py::class_<BasicAdoptor, SourceAdoptor, std::shared_ptr<BasicAdoptor>>(
          m, "BasicAdoptor", py::module_local());

  auto _MMapAdoptor =
      py::class_<MMapAdoptor, SourceAdoptor, std::shared_ptr<MMapAdoptor>>(
          m, "MMapAdoptor", py::module_local());

  auto _BytesAdoptor =
      py::class_<BytesAdoptor, SourceAdoptor, std::shared_ptr<BytesAdoptor>>(
          m, "BytesAdoptor", py::module_local());

  _SourceAdoptor.def("get", &SourceAdoptor::get)
      .def("__repr__", [](const SourceAdoptor& _) -> std::string {
        return "SourceAdoptor";
      });

  _BasicAdoptor
      .def(
          py::init<const std::optional<std::string>&>(),
          py::arg("prefix") = py::none())
      .def("__repr__", [](const BasicAdoptor& self) -> std::string {
        return fmt::format(
            "BasicAdoptor(prefix=\"{}\")", self.prefix.value_or(""));
      });

  _MMapAdoptor
      .def(
          py::init<const std::optional<std::string>&>(),
          py::arg("prefix") = py::none())
      .def("__repr__", [](const MMapAdoptor& self) -> std::string {
        return fmt::format(
            "MMapAdoptor(prefix=\"{}\")", self.prefix.value_or(""));
      });

  _BytesAdoptor
      .def(
          py::init<const std::optional<std::string>&>(),
          py::arg("prefix") = py::none())
      .def("__repr__", [](const BytesAdoptor& self) -> std::string {
        return fmt::format(
            "BytesAdoptor(prefix=\"{}\")", self.prefix.value_or(""));
      });
}

} // namespace spdl::core
