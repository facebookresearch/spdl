/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/storage.h>

#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#ifdef SPDL_USE_NVCODEC
#define _RET_ATTR
#else
// To workaround -Wmissing-noreturn when NVDEC is disabled
#define _RET_ATTR [[noreturn]]
#endif

namespace spdl::cuda {

namespace detail {
class NvDecDecoderCore;
}

/// Generator type for yielding decoded frame batches.
using CUDABufferGenerator = spdl::core::Generator<CUDABuffer>;

/// NVDEC-based video decoder for hardware-accelerated video decoding.
///
/// This class provides two decoding modes:
///
/// 1. **Batch decoding** - Decode all packets at once:
/// @code
///   NvDecDecoder decoder;
///   decoder.init_decoder(cuda_config, codec, crop, width, height);
///   auto buffer = decoder.decode_packets(packets);
/// @endcode
///
/// 2. **Streaming decoding** - Process packets incrementally with batched
/// output:
/// @code
///   NvDecDecoder decoder;
///   decoder.init_decoder(cuda_config, codec, crop, width, height);
///   decoder.init_buffer(num_frames);  // Initialize frame buffer
///
///   for (auto packets : packet_stream) {
///     for (auto batch : decoder.streaming_decode_packets(packets)) {
///       // Process batch (CUDABuffer with num_frames)
///     }
///   }
///
///   // Flush and get remaining frames
///   for (auto batch : decoder.flush()) {
///     // Process final batches
///   }
/// @endcode
///
/// **Error Recovery**:
/// If an error occurs, call reset() before reinitializing:
/// @code
///   decoder.reset();
///   decoder.init_decoder(cuda_config, codec, ...);
/// @endcode
class NvDecDecoder {
#ifdef SPDL_USE_NVCODEC
  detail::NvDecDecoderCore* core_;
#endif

 public:
  NvDecDecoder();
  NvDecDecoder(const NvDecDecoder&) = delete;
  NvDecDecoder& operator=(const NvDecDecoder&) = delete;
  // Deleting the move constructor for now.
  NvDecDecoder(NvDecDecoder&&) = delete;
  NvDecDecoder& operator=(NvDecDecoder&&) = delete;

  ~NvDecDecoder();

  _RET_ATTR void reset();

  // Initialize the decoder object for a new stream of
  // video packets
  _RET_ATTR void init_decoder(
      // device config
      const CUDAConfig& cuda_config,
      // Source codec information
      const spdl::core::VideoCodec& codec,
      // Post-decoding processing params
      CropArea crop,
      int width = -1,
      int height = -1);

  ////////////////////////////////////////////////////////////////////////////
  // One-off decoding
  ////////////////////////////////////////////////////////////////////////////

  // Decode all packets and return NV12 buffer.
  // Allocates the buffer internally based on packet count.
  // The buffer shape's first dimension reflects the actual frame count.
  _RET_ATTR CUDABuffer decode_packets(spdl::core::VideoPacketsPtr packets);

  ////////////////////////////////////////////////////////////////////////////
  // Streaming decoding
  ////////////////////////////////////////////////////////////////////////////

  // Initialize frame buffer for streaming decode
  _RET_ATTR void init_buffer(size_t num_frames);

  // Streaming decode packets and yield batches.
  //
  // This method decodes packets and yields batches of frames as they become
  // ready. The frame buffer must be initialized with init_buffer() before
  // calling this.
  //
  // @param packets Video packets to decode.
  // @return Generator that yields CUDABuffer batches in NV12 format.
  _RET_ATTR CUDABufferGenerator
  streaming_decode_packets(spdl::core::VideoPacketsPtr packets);

  // Flush the decoder and yield remaining batches.
  //
  // Call this method at the end of video stream to flush the decoder
  // and retrieve any remaining buffered frames as batches.
  //
  // @return Generator that yields remaining CUDABuffer batches in NV12 format.
  _RET_ATTR CUDABufferGenerator flush();
};

} // namespace spdl::cuda

#undef _RET_ATTR
