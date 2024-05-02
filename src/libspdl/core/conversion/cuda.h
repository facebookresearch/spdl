#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/conversion.h>

#include <functional>

namespace spdl::core {
BufferPtr convert_to_cuda(
    BufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator,
    bool async = false);
} // namespace spdl::core
