/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/packets.h>
#include <libspdl/cuda/buffer.h>

#include "libspdl/cuda/nvdec/detail/wrapper.h"

#include <cuda.h>
#include <nvcuvid.h>

#include <vector>

namespace spdl::cuda::detail {

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
class NvDecDecoderCore2 {
  //---------------------------------------------------------------------------
  // Device config
  // used to check if decoder configuration needs to be updated.
  // or allocating new memory
  CUDAConfig device_config;

  //---------------------------------------------------------------------------
  // Codec config
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
  CUvideoparserPtr parser{nullptr};
  CUvideodecoderPtr decoder{nullptr};

 private:
  int src_width = 0;
  int src_height = 0;
  // Resize option Negative values mean not resizing.
  int target_width = -1;
  int target_height = -1;
  // Cropping options
  CropArea crop;

  // Used to disable all the callbacks just in case.
  bool cb_disabled = false;

  // Storage for the output frames
  // Used as a reference point for the callback during the decoding.
  // This is only valid during decoding a packets.
  // (during the duration of `decode_packets` method.)
  // Do not access it from other locations.
  std::vector<CUDABuffer>* frame_buffer;

 public:
  NvDecDecoderCore2() = default;
  NvDecDecoderCore2(const NvDecDecoderCore2&) = delete;
  NvDecDecoderCore2& operator=(const NvDecDecoderCore2&) = delete;
  NvDecDecoderCore2(NvDecDecoderCore2&&) = delete;
  NvDecDecoderCore2& operator=(NvDecDecoderCore2&&) noexcept = delete;
  ~NvDecDecoderCore2() = default;

  // Reset the state of decoder
  void reset();

  // Needs to be called before the start of new decoding stream.
  void init(
      const CUDAConfig& device_config,
      const spdl::core::VideoCodec& codec,
      const CropArea& crop,
      int target_width = -1,
      int target_height = -1);

 private:
  // Decode packets using `cuvidParseVideoData`. It synchronously triggers
  // callbacks.
  void decode_packet(
      const uint8_t* data,
      const uint size,
      int64_t pts,
      unsigned long flags);

 public:
  void decode_packets(
      spdl::core::VideoPackets* packets,
      std::vector<CUDABuffer>* buffer);

  // Same as `decode` but notify the decode of the end of the stream.
  void flush(std::vector<CUDABuffer>* buffer);

  ////////////////////////////////////////////////////////////////////////////
  // Callbacks required by CUVID API
  ////////////////////////////////////////////////////////////////////////////
  int handle_video_sequence(CUVIDEOFORMAT*);
  int handle_decode_picture(CUVIDPICPARAMS*);
  int handle_display_picture(CUVIDPARSERDISPINFO*);
  int handle_operating_point(CUVIDOPERATINGPOINTINFO*);
  int handle_sei_msg(CUVIDSEIMESSAGEINFO*);
};

} // namespace spdl::cuda::detail
