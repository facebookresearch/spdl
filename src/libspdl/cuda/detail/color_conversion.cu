/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/detail/color_conversion.h"
#include "libspdl/cuda/utils.h"

#include <cstdint>

namespace spdl::cuda::detail {
namespace {
// YUV to RGB conversion matrix.
// They correspond to
// CUVIDEOFORMAT::video_signal_description.matrix_coefficients
__constant__ __device__ float yuv2rgb[10][3][3] = {
    // 1. BT709
    {{1.1644, 0.0000, 1.8337},
     {1.1644, -0.2181, -0.5451},
     {1.1644, 2.1606, 0.0000}},

    // 2. Unspecified
    {{1.1644, 0.0000, 1.8337},
     {1.1644, -0.2181, -0.5451},
     {1.1644, 2.1606, 0.0000}},

    // 3. Reserved
    {{1.1644, 0.0000, 1.8337},
     {1.1644, -0.2181, -0.5451},
     {1.1644, 2.1606, 0.0000}},

    // 4. FCC
    {{1.1644, 0.0000, 1.6301},
     {1.1644, -0.3864, -0.8289},
     {1.1644, 2.0726, 0.0000}},

    // 5. BT470
    {{1.1644, 0.0000, 1.6325},
     {1.1644, -0.4007, -0.8315},
     {1.1644, 2.0633, 0.0000}},

    // 6. BT601
    {{1.1644, 0.0000, 1.6325},
     {1.1644, -0.4007, -0.8315},
     {1.1644, 2.0633, 0.0000}},

    // 7. SMPTE240M
    {{1.1644, 0.0000, 1.8351},
     {1.1644, -0.2639, -0.5550},
     {1.1644, 2.1262, 0.0000}},

    // 8. YCgCo
    {{1.1644, 0.0000, 1.8337},
     {1.1644, -0.2181, -0.5451},
     {1.1644, 2.1606, 0.0000}},

    // 9. BT2020
    {{1.1689, 0.0000, 1.7237},
     {1.1689, -0.1924, -0.6679},
     {1.1689, 2.1992, 0.0000}},

    // 10. BT2020C
    {{1.1689, 0.0000, 1.7237},
     {1.1689, -0.1924, -0.6679},
     {1.1689, 2.1992, 0.0000}},
};

__device__ static uint8_t clamp8(float x) {
  return x < 0.0f ? 0 : (x > 255.0f ? 255 : static_cast<uint8_t>(x));
}

template <class COLOR24>
__device__ inline COLOR24
yuv_to_rgb_pixel(uint8_t y, uint8_t u, uint8_t v, float mat[3][3]) {
  const int low = 16, mid = 128;
  float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
  COLOR24 rgb{};
  rgb.c.r = clamp8(mat[0][0] * fy + mat[0][1] * fu + mat[0][2] * fv);
  rgb.c.g = clamp8(mat[1][0] * fy + mat[1][1] * fu + mat[1][2] * fv);
  rgb.c.b = clamp8(mat[2][0] * fy + mat[2][1] * fu + mat[2][2] * fv);
  return rgb;
}

template <class COLOR24>
__global__ static void nv12_to_planar_rgb24(
    uint8_t* yuv_ptr,
    int yuv_pitch,
    uint8_t* rgb_ptr,
    int rgb_pitch,
    int width,
    int height,
    int matrix_coefficients) {
  int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  if (x + 1 >= width || y + 1 >= height) {
    return;
  }

  uint8_t* y00 = yuv_ptr + y * yuv_pitch + x;
  uint8_t* y01 = y00 + 1;
  uint8_t* y10 = y00 + yuv_pitch;
  uint8_t* y11 = y10 + 1;

  uint8_t* u = yuv_ptr + ((height + y / 2) * yuv_pitch) + x;
  uint8_t* v = u + 1;

  if (matrix_coefficients <= 0 || 10 < matrix_coefficients) {
    matrix_coefficients = 1;
  }
  auto mat = yuv2rgb[matrix_coefficients - 1];

  auto rgb00 = yuv_to_rgb_pixel<COLOR24>(*y00, *u, *v, mat),
       rgb01 = yuv_to_rgb_pixel<COLOR24>(*y01, *u, *v, mat),
       rgb10 = yuv_to_rgb_pixel<COLOR24>(*y10, *u, *v, mat),
       rgb11 = yuv_to_rgb_pixel<COLOR24>(*y11, *u, *v, mat);

  rgb_ptr += x + y * rgb_pitch;
  for (int i = 0; i < 3; ++i) {
    *rgb_ptr = rgb00.v[i];
    *(rgb_ptr + 1) = rgb01.v[i];
    *(rgb_ptr + rgb_pitch) = rgb10.v[i];
    *(rgb_ptr + rgb_pitch + 1) = rgb11.v[i];
    rgb_ptr += height * rgb_pitch;
  }
}

union BGR24 {
  uint8_t v[3];
  struct {
    uint8_t b, g, r;
  } c;
};

union RGB24 {
  uint8_t v[3];
  struct {
    uint8_t r, g, b;
  } c;
};
} // namespace

void nv12_to_planar_rgb(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients) {
  auto dimGrid = dim3((width + 63) / 64, (height + 3) / 4);
  auto dimBlock = dim3(32, 2);
  TRACE_EVENT("nvdec", "nv12_to_planar_rgb");
  nv12_to_planar_rgb24<RGB24><<<dimGrid, dimBlock, 0, stream>>>(
      src, src_pitch, dst, dst_pitch, width, height, matrix_coefficients);
  CHECK_CUDA(
      cudaPeekAtLastError(),
      "Failed to launch kernel nv12_to_planar_rgb<RGB24>");
}

void nv12_to_planar_bgr(
    CUstream stream,
    uint8_t* src,
    int src_pitch,
    uint8_t* dst,
    int dst_pitch,
    int width,
    int height,
    int matrix_coefficients) {
  auto dimGrid = dim3((width + 63) / 64, (height + 3) / 4);
  auto dimBlock = dim3(32, 2);
  TRACE_EVENT("nvdec", "nv12_to_planar_bgra");
  nv12_to_planar_rgb24<BGR24><<<dimGrid, dimBlock, 0, stream>>>(
      src, src_pitch, dst, dst_pitch, width, height, matrix_coefficients);
  CHECK_CUDA(
      cudaPeekAtLastError(),
      "Failed to launch kernel nv12_to_planar_bgr<BGR24>");
}
} // namespace spdl::cuda::detail
