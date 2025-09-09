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
class NvDecDecoderCore {
  //---------------------------------------------------------------------------
  // Device config
  //---------------------------------------------------------------------------
  // used to check if decoder configuration needs to be updated.
  // or allocating new memory
  CUDAConfig device_config{
      .device_index = -1
      // Initialize with invalid index, otherwise when `init` is called, it
      // cannot tell whether it's genuinely initialized.
  };
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // Codec config
  //---------------------------------------------------------------------------
  // Codec associated with the parser
  cudaVideoCodec codec{};
  // Params from previous decoder creation.
  CUVIDDECODECREATEINFO decoder_param{};
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // Global objects (handle to device)
  //---------------------------------------------------------------------------
  CUcontext cu_ctx;
  CUvideoctxlock lock;
  //---------------------------------------------------------------------------

  // Cache the result of `cuvidGetDecoderCaps` as
  // `cuvidGetDecoderCaps` is expensive.
  std::vector<CUVIDDECODECAPS> cap_cache{};

  // Core decoder objects
  CUvideoparserPtr parser{nullptr};
  CUvideodecoderPtr decoder{nullptr};

 private:
  // Source packet information. Initialized in init
  int src_width = 0;
  int src_height = 0;
  spdl::core::CodecID codec_id;
  spdl::core::Rational timebase{}; // Time base of the PTS

  //---------------------------------------------------------------------------
  // Post processing params
  //---------------------------------------------------------------------------
  // Resize option Negative values mean not resizing.
  int target_width = -1;
  int target_height = -1;
  // Cropping options
  CropArea crop;
  //---------------------------------------------------------------------------

  // Used to disable all the callbacks during reset.
  bool cb_disabled = false;

  //---------------------------------------------------------------------------
  // Attributes used for decoding, only during the decoding.
  // Should be set at the beginning of decoding, and should be only accessed
  // from callbacks.
  //---------------------------------------------------------------------------
  // Storage for the output frames
  // Used as a reference point for the callback during the decoding.
  std::vector<CUDABuffer>* frame_buffer;
  // The user-specified timestamp. Frames outside of this will be discarded.
  double start_time, end_time;
  //---------------------------------------------------------------------------

 public:
  NvDecDecoderCore() = default;
  NvDecDecoderCore(const NvDecDecoderCore&) = delete;
  NvDecDecoderCore& operator=(const NvDecDecoderCore&) = delete;
  NvDecDecoderCore(NvDecDecoderCore&&) = delete;
  NvDecDecoderCore& operator=(NvDecDecoderCore&&) noexcept = delete;
  ~NvDecDecoderCore() = default;

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
      const unsigned long size,
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
