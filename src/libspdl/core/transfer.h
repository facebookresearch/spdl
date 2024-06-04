#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>

#include <vector>

namespace spdl::core {

CUDABufferPtr transfer_buffer(CPUBufferPtr buffer, const CUDAConfig& cfg);

CPUStorage cp_to_cpu(const void* src, const std::vector<size_t> shape);

} // namespace spdl::core
