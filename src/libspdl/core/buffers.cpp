#include <libspdl/core/buffers.h>

#include <libspdl/core/logging.h>

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(
    const std::vector<size_t> shape_,
    bool channel_last_,
    ElemClass elem_class_,
    size_t depth_,
    Storage&& storage_)
    : shape(std::move(shape_)),
      elem_class(elem_class_),
      depth(depth_),
      channel_last(channel_last_),
      storage(std::make_shared<StorageVariants>(std::move(storage_))) {}

#ifdef SPDL_USE_CUDA
Buffer::Buffer(
    const std::vector<size_t> shape_,
    bool channel_last_,
    CUDAStorage&& storage_)
    : shape(std::move(shape_)),
      channel_last(channel_last_),
      storage(std::make_shared<StorageVariants>(std::move(storage_))) {}
#endif

template <class... Fs>
struct Overload : Fs... {
  using Fs::operator()...;
};
template <class... Fs>
Overload(Fs...) -> Overload<Fs...>;

void* Buffer::data() {
  return std::visit(
      Overload{
          [](Storage& s) { return s.data; }
#ifdef SPDL_USE_CUDA
          ,
          [](CUDAStorage& s) { return s.data; }
#endif
      },
      *storage);
}

bool Buffer::is_cuda() const {
#ifdef SPDL_USE_CUDA
  assert(storage);
  return std::holds_alternative<CUDAStorage>(*storage);
#else
  return false;
#endif
}

#ifdef SPDL_USE_CUDA
uintptr_t Buffer::get_cuda_stream() const {
  if (!std::holds_alternative<CUDAStorage>(*storage)) {
    SPDL_FAIL_INTERNAL("CUDAStream is not available.");
  }
  return (uintptr_t)std::get<CUDAStorage>(*storage).stream;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////
namespace {
inline size_t prod(const std::vector<size_t>& shape) {
  size_t val = 1;
  for (auto& v : shape) {
    val *= v;
  }
  return val;
}
} // namespace

Buffer cpu_buffer(
    const std::vector<size_t> shape,
    bool channel_last,
    ElemClass elem_class,
    size_t depth) {
  XLOG(DBG) << fmt::format(
      "Allocating {} bytes. (shape: {}, elem: {})",
      prod(shape) * depth,
      fmt::join(shape, ", "),
      depth);
  return Buffer{
      std::move(shape),
      channel_last,
      elem_class,
      depth,
      Storage{prod(shape) * depth}};
}

#ifdef SPDL_USE_CUDA
Buffer cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last) {
  return Buffer{
      std::move(shape), channel_last, CUDAStorage{prod(shape), stream}};
}
#endif

} // namespace spdl::core
