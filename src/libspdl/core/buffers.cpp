#include <libspdl/core/buffers.h>

#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#ifdef SPDL_USE_NVDEC
#include <libspdl/core/detail/cuda.h>
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

#ifdef SPDL_USE_NVDEC
CUDABuffer2DPitch::CUDABuffer2DPitch(size_t max_frames_)
    : max_frames(max_frames_) {}

CUDABuffer2DPitch::~CUDABuffer2DPitch() {
  if (p) {
    TRACE_EVENT("nvdec", "cuMemFree");
    CHECK_CU(cuMemFree(p), "Failed to free memory.");
  }
}

void CUDABuffer2DPitch::allocate(
    size_t c_,
    size_t h_,
    size_t w_,
    size_t bpp_,
    bool channel_last_) {
  if (p) {
    SPDL_FAIL_INTERNAL("Arena is already allocated.");
  }
  channel_last = channel_last_;
  c = c_, h = h_, w = w_, bpp = bpp_;

  width_in_bytes = channel_last ? w * c * bpp : w * bpp;
  size_t height = channel_last ? max_frames * h : max_frames * c * h;

  TRACE_EVENT("nvdec", "cuMemAllocPitch");
  CHECK_CU(
      cuMemAllocPitch((CUdeviceptr*)&p, &pitch, width_in_bytes, height, 8),
      "Failed to allocate memory.");
}

std::vector<size_t> CUDABuffer2DPitch::get_shape() const {
  return channel_last ? std::vector<size_t>{n, h, w, c}
                      : std::vector<size_t>{n, c, h, w};
}

uint8_t* CUDABuffer2DPitch::get_next_frame() {
  if (!p) {
    SPDL_FAIL_INTERNAL("Memory is not allocated.");
  }
  if (n >= max_frames) {
    SPDL_FAIL_INTERNAL(
        "Attempted to write beyond the maximum number of frames.");
  }
  return channel_last ? (uint8_t*)p + n * h * pitch
                      : (uint8_t*)p + n * c * h * pitch;
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
      new CPUStorage{prod(shape) * depth});
}

#ifdef SPDL_USE_CUDA
std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last) {
  return std::make_unique<CUDABuffer>(
      std::move(shape),
      channel_last,
      ElemClass::UInt,
      sizeof(uint8_t),
      new CUDAStorage{prod(shape), stream});
}
#endif

} // namespace spdl::core
