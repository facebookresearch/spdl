#pragma once

#include "libspdl/core/detail/nvdec/buffer.h"

#include <cuviddec.h>

namespace spdl::core::detail {

struct Converter {
  CUstream stream;
  CUDABufferTracker* tracker;

  Converter(CUstream stream, CUDABufferTracker* tracker);

  virtual ~Converter() = default;

  /// @param src_ptr CUdeviceptr
  /// @param src_pitch Pitch in bytes
  virtual void convert(uint8_t* src_ptr, unsigned int src_pitch) = 0;
};

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABufferTracker* tracker,
    const CUVIDDECODECREATEINFO* decoder_param,
    unsigned char matrix_coeff,
    const std::optional<std::string>& pix_fmt);

} // namespace spdl::core::detail
