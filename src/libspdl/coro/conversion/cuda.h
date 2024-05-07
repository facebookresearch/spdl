#pragma once

#include <libspdl/coro/conversion.h>

#include <libspdl/core/buffer.h>

#include <functional>

namespace spdl::core {
BufferPtr convert_to_cuda(
    BufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator,
    bool async = false);
} // namespace spdl::core
