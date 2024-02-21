#pragma once

#include <libspdl/core/buffers.h>

#include <cuviddec.h>

namespace spdl::core::detail {

struct Converter {
  CUstream stream;
  CUDABuffer2DPitch* buffer;

  Converter(CUstream stream, CUDABuffer2DPitch* buffer);

  /// @param src_ptr CUdeviceptr
  /// @param src_pitch Pitch in bytes
  virtual ~Converter() = default;
  virtual void convert(uint8_t* src_ptr, unsigned int src_pitch) = 0;
};

class NV12Passthrough : public Converter {
 public:
  using Converter::Converter;
  void convert(uint8_t* src_ptr, unsigned int src_pitch) override;
};

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABuffer2DPitch* buffer,
    const CUVIDDECODECREATEINFO* decoder_param);

} // namespace spdl::core::detail
