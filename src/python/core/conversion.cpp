#include <libspdl/core/conversion.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
CPUBufferPtr convert(const FFmpegFramesPtr<media_type> frames) {
  nb::gil_scoped_release g;
  return convert_frames(frames.get());
}

template <MediaType media_type>
std::vector<const spdl::core::FFmpegFrames<media_type>*> _ref(
    std::vector<FFmpegFramesPtr<media_type>>& frames) {
  std::vector<const spdl::core::FFmpegFrames<media_type>*> ret;
  for (auto& frame : frames) {
    ret.push_back(frame.get());
  }
  return ret;
}

template <MediaType media_type>
CPUBufferPtr batch_convert(std::vector<FFmpegFramesPtr<media_type>>&& frames) {
  nb::gil_scoped_release g;
  return convert_frames(_ref(frames));
}

CUDABufferPtr _transfer_to_cuda(
    CPUBufferPtr buffer,
    const TransferConfig& cfg) {
  nb::gil_scoped_release g;
  return convert_to_cuda(std::move(buffer), cfg);
}

template <typename IntType = int32_t>
CPUBufferPtr _convert_tokens_1d(
    const nb::list tokens,
    std::optional<size_t> token_length) {
  auto num_tokens = tokens.size();
  if (num_tokens == 0) {
    throw std::runtime_error("Token is empty");
  }
  if (token_length && *token_length == 0) {
    throw std::runtime_error("`token_length` cannot be zero.");
  }
  auto len = token_length ? *token_length : num_tokens;
  auto buffer = cpu_buffer({len}, ElemClass::Int, sizeof(IntType));
  auto data = static_cast<IntType*>(buffer->data());
  for (size_t i = 0; i < std::min(len, num_tokens); ++i) {
    *data = nb::cast<IntType>(tokens[i]);
    ++data;
  }
  if (len > num_tokens) {
    std::memset(data, 0, sizeof(IntType) * (len - num_tokens));
  }
  return buffer;
}

template <typename IntType = int32_t>
CPUBufferPtr _convert_tokens_2d(
    const nb::list batch_tokens,
    std::optional<size_t> token_length) {
  auto num_batch = batch_tokens.size();
  if (num_batch == 0) {
    throw std::runtime_error("Batch is empty");
  }

  auto max_token_length = [&]() -> size_t {
    size_t ret = 0;
    for (nb::handle h : batch_tokens) {
      ret = std::max(ret, nb::cast<nb::list>(h).size());
    }
    return ret;
  }();
  if (max_token_length == 0) {
    throw std::runtime_error("Tokens are empty");
  }

  auto len = token_length ? *token_length : max_token_length;
  auto buffer = cpu_buffer({num_batch, len}, ElemClass::Int, sizeof(IntType));
  auto data = static_cast<IntType*>(buffer->data());
  for (nb::handle h : batch_tokens) {
    auto tokens = nb::cast<nb::list>(h);
    auto num_tokens = tokens.size();
    for (size_t i = 0; i < std::min(len, num_tokens); ++i) {
      *data = nb::cast<IntType>(tokens[i]);
      ++data;
    }
    if (len > num_tokens) {
      std::memset(data, 0, sizeof(IntType) * (len - num_tokens));
      data += len - num_tokens;
    }
  }
  return buffer;
}

} // namespace

void register_conversion(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Frame conversion
  ////////////////////////////////////////////////////////////////////////////////
  m.def("convert_frames", &convert<MediaType::Audio>, nb::arg("frames"));
  m.def("convert_frames", &convert<MediaType::Video>, nb::arg("frames"));
  m.def("convert_frames", &convert<MediaType::Image>, nb::arg("frames"));

  m.def("convert_frames", &batch_convert<MediaType::Audio>, nb::arg("frames"));
  m.def("convert_frames", &batch_convert<MediaType::Video>, nb::arg("frames"));
  m.def("convert_frames", &batch_convert<MediaType::Image>, nb::arg("frames"));

  ////////////////////////////////////////////////////////////////////////////////
  // Device trancfer
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "transfer_to_cuda",
      &_transfer_to_cuda,
      nb::arg("buffer"),
      nb::kw_only(),
      nb::arg("transfer_config"));

  ////////////////////////////////////////////////////////////////////////////////
  // Convert list of integers (tokens)
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_tokens_1d",
      &_convert_tokens_1d<>,
      nb::arg("tokens"),
      nb::arg("token_length") = nb::none());

  m.def(
      "convert_tokens_2d",
      &_convert_tokens_2d<>,
      nb::arg("tokens"),
      nb::arg("token_length") = nb::none());
}
} // namespace spdl::core
