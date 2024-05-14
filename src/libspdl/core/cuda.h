#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/storage.h>

namespace spdl::core {
CUDABufferPtr convert_to_cuda(
    CPUBufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator,
    bool async = false);

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t> shape);

} // namespace spdl::core
