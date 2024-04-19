#include <libspdl/core/buffer.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// CUDABuffer
////////////////////////////////////////////////////////////////////////////////
CUDABuffer::CUDABuffer(
    std::vector<size_t> shape_,
    bool channel_last_,
    ElemClass elem_class_,
    size_t depth_,
    CUDAStorage* storage_,
    int device_index_)
    : Buffer(
          std::move(shape_),
          channel_last_,
          elem_class_,
          depth_,
          (Storage*)storage_),
      device_index(device_index_) {}

uintptr_t CUDABuffer::get_cuda_stream() const {
  return (uintptr_t)(((CUDAStorage*)(storage.get()))->stream);
}

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

std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    int device_index,
    bool channel_last,
    ElemClass elem_class,
    size_t depth) {
  return std::make_unique<CUDABuffer>(
      std::move(shape),
      channel_last,
      elem_class,
      depth,
      new CUDAStorage{depth * prod(shape), stream},
      device_index);
}

std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    uintptr_t stream,
    int device_index,
    bool channel_last,
    ElemClass elem_class,
    size_t depth,
    const cuda_allocator_fn& allocator,
    cuda_deleter_fn deleter) {
  return std::make_unique<CUDABuffer>(
      std::move(shape),
      channel_last,
      elem_class,
      depth,
      new CUDAStorage{
          depth * prod(shape), device_index, stream, allocator, deleter},
      device_index);
}

} // namespace spdl::core
