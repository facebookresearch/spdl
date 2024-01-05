#include <libspdl/buffers.h>

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl {

Buffer::Buffer(
    const std::vector<size_t> shape_,
    bool channel_last_,
    Storage&& storage_)
    : shape(std::move(shape_)),
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
          [](Storage& s) { return static_cast<void*>(s.data); }
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

namespace {

inline size_t prod(const std::vector<size_t>& shape) {
  size_t val = 1;
  for (auto& v : shape) {
    val *= v;
  }
  return val;
}

} // namespace

Buffer cpu_buffer(const std::vector<size_t> shape, bool channel_last) {
  return Buffer{std::move(shape), channel_last, Storage{prod(shape)}};
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

} // namespace spdl
