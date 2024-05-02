#include <libspdl/core/buffer.h>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// CUDABuffer
////////////////////////////////////////////////////////////////////////////////
CUDABuffer::CUDABuffer(
    std::vector<size_t> shape_,
    ElemClass elem_class_,
    size_t depth_,
    CUDAStorage* storage_,
    int device_index_)
    : Buffer(std::move(shape_), elem_class_, depth_, (Storage*)storage_),
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

CUDABufferPtr cuda_buffer(
    const std::vector<size_t> shape,
    int device_index,
    uintptr_t stream,
    const std::optional<cuda_allocator>& allocator,
    ElemClass elem_class,
    size_t depth) {
  return std::make_unique<CUDABuffer>(
      std::move(shape),
      elem_class,
      depth,
      new CUDAStorage{depth * prod(shape), device_index, stream, allocator},
      device_index);
}

} // namespace spdl::core
