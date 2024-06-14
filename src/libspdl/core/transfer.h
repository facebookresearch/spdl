#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>

#include <vector>

namespace spdl::core {

// CPU -> CUDA
CUDABufferPtr transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg);
CUDABufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg);

// CUDA -> CPU
CPUBufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    ElemClass elem_class,
    size_t depth,
    const void* ptr);

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t>& shape);
} // namespace spdl::core
