/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/types.h>

#include <string>
#include <vector>

namespace spdl::core {

void encode_image(
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    int bit_depth,
    const std::string& src_pix_fmt,
    const std::optional<EncodeConfig>& enc_cfg);

} // namespace spdl::core
