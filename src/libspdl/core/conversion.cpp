#include <libspdl/core/conversion.h>

#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#endif
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
namespace {
size_t prod(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto& v : shape) {
    ret *= v;
  }
  return ret;
}

} // namespace

CUDABufferPtr convert_to_cuda(CPUBufferPtr buffer, const TransferConfig& cfg) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  TRACE_EVENT("decoding", "core::convert_to_cuda");

  auto ret = cuda_buffer(
      buffer->shape,
      cfg.device_index,
      cfg.stream,
      cfg.allocator,
      buffer->elem_class,
      buffer->depth);

  size_t size = buffer->depth * prod(buffer->shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), buffer->data(), size, cudaMemcpyHostToDevice),
      "Failed to copy data from host to device.");

  return ret;
#endif
}

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t> shape) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else

  size_t size = prod(shape);
  CPUStorage storage{size};

  CHECK_CUDA(
      cudaMemcpy(storage.data(), src, size, cudaMemcpyDeviceToHost),
      "Failed to copy data from device to host.");

  return storage;
#endif
}

} // namespace spdl::core
