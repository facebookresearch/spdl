/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <nvcuvid.h>

namespace spdl::core::detail {

struct CUvideoparserDeleter {
  void operator()(CUvideoparser p);
};

using CUvideoparserPtr =
    std::unique_ptr</* *CUvideoparser */ void, CUvideoparserDeleter>;

struct CUvideodecoderDeleter {
  void operator()(void* p);
};

using CUvideodecoderPtr =
    std::unique_ptr</* *CUvideodecoder */ void, CUvideodecoderDeleter>;

struct CUvideoctxlockDeleter {
  void operator()(void* p);
};

using CUvideoctxlockPtr =
    std::unique_ptr</* *CUvideoctxlock */ void, CUvideoctxlockDeleter>;

// Perform cuvidMapVideoFrame/cuvidUnmapVideoFrame in RAII manner
struct MapGuard {
  CUvideodecoder decoder;

  CUdeviceptr frame = 0;
  unsigned int pitch = 0;

  MapGuard(
      CUvideodecoder decoder,
      CUVIDPROCPARAMS* proc_params,
      int picture_index);
  ~MapGuard();

  MapGuard(const MapGuard&) = delete;
  MapGuard& operator=(const MapGuard&) = delete;
  MapGuard(MapGuard&&) noexcept = delete;
  MapGuard& operator=(MapGuard&&) noexcept = delete;
};

} // namespace spdl::core::detail
