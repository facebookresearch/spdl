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
typedef CUstream_st* CUstream;

namespace spdl::core::detail {

void nv12_to_planar_rgba(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients);

void nv12_to_planar_bgra(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients);

} // namespace spdl::core::detail
