/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/packets.h>

#include <libspdl/core/adaptor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace spdl::core::detail {

template <MediaType media_type>
CUDABufferPtr decode_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width = -1,
    int target_height = -1,
    const std::optional<std::string>& pix_fmt = std::nullopt);

} // namespace spdl::core::detail
