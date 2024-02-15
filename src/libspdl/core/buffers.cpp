#include <libspdl/core/buffers.h>

#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#ifdef SPDL_USE_CUDA
#include <libspdl/core/detail/cuda.h>

#include <cuda_runtime.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(
    std::vector<size_t> shape_,
    bool channel_last_,
    ElemClass elem_class_,
    size_t depth_,
    Storage* storage_)
    : shape(std::move(shape_)),
      elem_class(elem_class_),
      depth(depth_),
      channel_last(channel_last_),
      storage(storage_) {}

void* Buffer::data() {
  return storage->data();
}

////////////////////////////////////////////////////////////////////////////////
// CPUBuffer
////////////////////////////////////////////////////////////////////////////////
CPUBuffer::CPUBuffer(
    std::vector<size_t> shape_,
    bool channel_last_,
    ElemClass elem_class_,
    size_t depth_,
    CPUStorage* storage_)
    : Buffer(
          std::move(shape_),
          channel_last_,
          elem_class_,
          depth_,
          (Storage*)storage_) {}

bool CPUBuffer::is_cuda() const {
  return false;
}

////////////////////////////////////////////////////////////////////////////////
// CUDABuffer
////////////////////////////////////////////////////////////////////////////////
#ifdef SPDL_USE_CUDA
CUDABuffer::CUDABuffer(
    std::vector<size_t> shape_,
    bool channel_last_,
    ElemClass elem_class_,
    size_t depth_,
    CUDAStorage* storage_)
    : Buffer(
          std::move(shape_),
          channel_last_,
          elem_class_,
          depth_,
          (Storage*)storage_) {}

bool CUDABuffer::is_cuda() const {
  return true;
}

uintptr_t CUDABuffer::get_cuda_stream() const {
  return (uintptr_t)(((CUDAStorage*)(storage.get()))->stream);
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

void* cpu_alloc(size_t size) {
  TRACE_EVENT("decoding", "operator new");
  return operator new(size);
}
} // namespace

std::unique_ptr<CPUBuffer> cpu_buffer(
    std::vector<size_t> shape,
    bool channel_last,
    ElemClass elem_class,
    size_t depth) {
  XLOG(DBG) << fmt::format(
      "Allocating {} bytes. (shape: {}, elem: {})",
      prod(shape) * depth,
      fmt::join(shape, ", "),
      depth);
  return std::make_unique<CPUBuffer>(
      std::move(shape),
      channel_last,
      elem_class,
      depth,
      new CPUStorage{cpu_alloc(prod(shape) * depth)});
}

#ifdef SPDL_USE_CUDA
namespace {
void* cuda_alloc(size_t size) {
  void* p;
  XLOG(DBG9) << fmt::format("Allocating CUDA memory ({} bytes)", size);
  TRACE_EVENT("decoding", "cudaMallocAsync");
  CHECK_CUDA(cudaMallocAsync(&p, size, 0), "Failed to allocate CUDA memory");
  XLOG(DBG9) << fmt::format("Allocation queued {}", p);
  return p;
}
} // namespace

std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last) {
  return std::make_unique<CUDABuffer>(
      std::move(shape),
      channel_last,
      ElemClass::UInt,
      sizeof(uint8_t),
      new CUDAStorage{cuda_alloc(prod(shape)), stream});
}
#endif

} // namespace spdl::core
