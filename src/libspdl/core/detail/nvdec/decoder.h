#pragma once

#include <libspdl/core/buffers.h>
#include <libspdl/core/detail/nvdec/utils.h>
#include <libspdl/core/detail/nvdec/wrapper.h>

#include <cuda.h>
#include <nvcuvid.h>

#include <optional>
#include <vector>

namespace spdl::core::detail {

// NOTE
// This class is designed to be used as thread local.
// This is because the creation/destruction of NVDEC decoder object
// (call to `cuvidCreateDecoder`/`cuvidDestroyDecoder`) is very expensive.
// It takes 300 ms, and voids the benefit of using NVDEC video decoder.
//
// Before the start of each job, `init` method must be called to
// re-configure the decoder. If the decoder is compatible with the
// previous job, then the decoder will be retained as-is or
// reconfigured using `cuvidReconfigureDecoder`.
//
// If the previous decoder configuration is not compatible with the new
// config, then the decoder object is re-created.
class NvDecDecoder {
  //---------------------------------------------------------------------------
  // Objects used to check if decoder configuration needs to be updated.
  // The device associated with the CUcontext.
  CUdevice device = -1;
  // Codec associated with the parser
  cudaVideoCodec codec{};
  // Params from previous decoder creation.
  CUVIDDECODECREATEINFO decoder_param{};
  //---------------------------------------------------------------------------

  // Global objects (handle to device)
  CUcontext cu_ctx;
  CUvideoctxlock lock;

  // Cache the result of `cuvidGetDecoderCaps` as
  // `cuvidGetDecoderCaps` is expensive.
  std::vector<CUVIDDECODECAPS> cap_cache{};

  // Core decoder objects
  CUstream stream = CU_STREAM_PER_THREAD;
  CUvideoparserPtr parser{nullptr};
  CUvideodecoderPtr decoder{nullptr};

  // Where the output frames will be stored.
  CUDABuffer2DPitch* buffer = nullptr;

  // Timebase of the incomding packets/decoded frames
  AVRational timebase;
  double start_time, end_time;

  // Resize option
  int target_width = -1;
  int target_height = -1;
  // Cropping options
  int crop_left = 0;
  int crop_top = 0;
  int crop_right = 0;
  int crop_bottom = 0;

 public:
  NvDecDecoder() = default;
  NvDecDecoder(const NvDecDecoder&) = delete;
  NvDecDecoder& operator=(const NvDecDecoder&) = delete;
  NvDecDecoder(NvDecDecoder&&) = delete;
  NvDecDecoder& operator=(NvDecDecoder&&) noexcept = delete;
  ~NvDecDecoder() = default;

  // Needs to be called before the start of new decoding stream.
  void init(
      CUdevice device_index,
      cudaVideoCodec codec,
      CUDABuffer2DPitch* buffer,
      AVRational timebase,
      std::tuple<double, double> timestamp,
      int crop_left = 0,
      int crop_top = 0,
      int crop_right = 0,
      int crop_bottom = 0,
      int target_width = -1,
      int target_height = -1);

  void decode(
      const uint8_t* data,
      const uint size,
      int64_t pts,
      unsigned long flags);

  ////////////////////////////////////////////////////////////////////////////
  // Callbacks required by CUVID API
  ////////////////////////////////////////////////////////////////////////////
  int handle_video_sequence(CUVIDEOFORMAT*);
  int handle_decode_picture(CUVIDPICPARAMS*);
  int handle_display_picture(CUVIDPARSERDISPINFO*);
  int handle_operating_point(CUVIDOPERATINGPOINTINFO*);
  int handle_sei_msg(CUVIDSEIMESSAGEINFO*);
};

} // namespace spdl::core::detail
