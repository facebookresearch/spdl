#pragma once

#include <libspdl/core/buffer.h>

#include <functional>

namespace spdl::core {
CUDABufferPtr convert_to_cuda(
    CPUBufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator,
    bool async = false);
} // namespace spdl::core
