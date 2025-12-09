/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <array>
#include <vector>

namespace spdl::core {

/// Convert a batch of frames to a CPU buffer.
///
/// Converts decoded frames into a contiguous CPU buffer suitable for
/// processing or copying to other memory systems.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param batch Vector of frame pointers to convert.
/// @param storage Optional pre-allocated storage. If not provided, allocates
/// new storage.
/// @return CPU buffer containing the converted frame data.
template <MediaType media>
CPUBufferPtr convert_frames(
    const std::vector<const Frames<media>*>& batch,
    std::shared_ptr<CPUStorage> storage = nullptr);

/// Convert a single frame to a CPU buffer.
///
/// Converts decoded frames into a contiguous CPU buffer suitable for
/// processing or copying to other memory systems.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param frames Frames to convert.
/// @param storage Optional pre-allocated storage. If not provided, allocates
/// new storage.
/// @return CPU buffer containing the converted frame data.
template <MediaType media>
CPUBufferPtr convert_frames(
    const Frames<media>* frames,
    std::shared_ptr<CPUStorage> storage = nullptr) {
  const std::vector<const Frames<media>*> batch{frames};
  // Use the same impl as batch conversion
  auto ret = convert_frames<media>(batch, std::move(storage));
  ret->shape.erase(ret->shape.begin()); // Trim the batch dim
  return ret;
}

/// Create an audio frame referencing external data.
///
/// Creates a frame that references externally managed memory without copying.
/// The caller must ensure the data remains valid for the lifetime of the frame.
///
/// @param sample_fmt Sample format string (e.g., "s16", "f32").
/// @param data Pointer to external audio data.
/// @param bits Bits per sample.
/// @param shape Shape array [num_channels, num_samples].
/// @param stride Stride array [channel_stride, sample_stride].
/// @param sample_rate Sample rate in Hz.
/// @param pts Presentation timestamp.
/// @return Audio frames referencing the external data.
AudioFramesPtr create_reference_audio_frame(
    const std::string& sample_fmt,
    const void* data,
    int bits,
    const std::array<size_t, 2>& shape,
    const std::array<int64_t, 2>& stride,
    int sample_rate,
    int64_t pts);

/// Create a video frame referencing external data.
///
/// Creates a frame that references externally managed memory without copying.
/// The caller must ensure the data remains valid for the lifetime of the frame.
///
/// @param sample_fmt Pixel format string (e.g., "rgb24", "yuv420p").
/// @param data Pointer to external video data.
/// @param bits Bits per component.
/// @param shape Shape vector [height, width, ...].
/// @param stride Stride vector for each dimension.
/// @param frame_rate Frame rate as a rational number.
/// @param pts Presentation timestamp.
/// @return Video frames referencing the external data.
VideoFramesPtr create_reference_video_frame(
    const std::string& sample_fmt,
    const void* data,
    int bits,
    const std::vector<size_t>& shape,
    const std::vector<int64_t>& stride,
    Rational frame_rate,
    int64_t pts);

} // namespace spdl::core
