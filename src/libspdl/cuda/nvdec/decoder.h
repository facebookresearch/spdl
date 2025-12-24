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

#include <libspdl/core/bsf.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

// Forward declaration
namespace spdl::core {
class Demuxer;
} // namespace spdl::core

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

// Usage:
// NvDecDecoder decoder{};
// decoder.init(...)
// for (auto packets : src) {
//   frames = decoder.decode(std::move(packets));
//   ...
// }
// frames = decoder.flush();
//
// If there was an error, then reset before
// initializing it again.
//
// decoder.reset();
// decoder.init();
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
  _RET_ATTR void init(
      // device config
      const CUDAConfig& cuda_config,
      // Source codec information
      const spdl::core::VideoCodec& codec,
      // Post-decoding processing params
      CropArea crop,
      int width = -1,
      int height = -1);

  // decode one stream of video packets
  _RET_ATTR std::vector<CUDABuffer> decode(spdl::core::VideoPacketsPtr packets);

  // Call this method at the end of video stream.
  _RET_ATTR std::vector<CUDABuffer> flush();

  // Decode all packets and return NV12 buffer.
  // Allocates the buffer internally based on packet count.
  // The buffer shape's first dimension reflects the actual frame count.
  _RET_ATTR CUDABuffer decode_all(spdl::core::VideoPacketsPtr packets);
};

/// Generator type for yielding batches of decoded frames.
using FrameBatchGenerator = spdl::core::Generator<std::vector<CUDABuffer>>;

/// Load video from source chunk by chunk using NVDEC.
///
/// This function demuxes video packets from the source, applies bitstream
/// filtering if needed (for H.264/HEVC), decodes using NVDEC, and yields
/// frames in batches.
///
/// @param demuxer Demuxer instance for the video source.
/// @param decoder Pre-initialized NvDecDecoder instance.
/// @param bsf Optional pre-initialized BSF instance for bitstream filtering.
/// @param num_frames Maximum number of frames to yield at a time.
/// @return Generator that yields batches of CUDABuffer frames in NV12 format.
_RET_ATTR FrameBatchGenerator streaming_load_video_nvdec(
    spdl::core::Demuxer* demuxer,
    NvDecDecoder* decoder,
    spdl::core::BSF<spdl::core::MediaType::Video>* bsf,
    size_t num_frames);

} // namespace spdl::cuda

#undef _RET_ATTR
