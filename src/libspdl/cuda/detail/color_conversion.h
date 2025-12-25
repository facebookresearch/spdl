/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

struct CUstream_st;
using CUstream = CUstream_st*;

namespace spdl::cuda::detail {

// Non-batched version (single frame)
void nv12_to_planar_rgb(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients);

void nv12_to_planar_bgr(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients);

// Batched version (multiple frames in single kernel launch)
void nv12_to_planar_rgb_batched(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int num_frames,
    int matrix_coefficients);

void nv12_to_planar_bgr_batched(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int num_frames,
    int matrix_coefficients);

} // namespace spdl::cuda::detail
