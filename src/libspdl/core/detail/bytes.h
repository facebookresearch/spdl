/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace spdl::core::detail {

int64_t seek_bytes(size_t size, size_t& position, int64_t offset, int whence);

} // namespace spdl::core::detail
