#pragma once

#include <libspdl/core/buffer.h>

namespace spdl::core {
BufferPtr convert_to_cuda(BufferPtr buffer, int cuda_device_index);
} // namespace spdl::core
