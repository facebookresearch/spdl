#pragma once

#include <libspdl/core/buffers.h>

#include <cuviddec.h>

namespace spdl::core::detail {

struct Converter {
  CUstream stream;
  CUDABuffer2DPitch* buffer;

  Converter(CUstream stream, CUDABuffer2DPitch* buffer);

  virtual ~Converter() = default;

  /// @param src_ptr CUdeviceptr
  /// @param src_pitch Pitch in bytes
  virtual void convert(uint8_t* src_ptr, unsigned int src_pitch) = 0;
};

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABuffer2DPitch* buffer,
    const CUVIDDECODECREATEINFO* decoder_param,
    unsigned char matrix_coeff,
    const std::optional<std::string>& pix_fmt);

} // namespace spdl::core::detail
