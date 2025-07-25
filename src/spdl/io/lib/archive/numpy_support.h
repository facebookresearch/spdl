/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace spdl::archive {

struct NPYArray {
  std::string descr{};
  bool fortran_order = false;
  std::vector<size_t> shape{};

  // Pointer to the array data (not owned)
  void* data = nullptr;

  // Owned data (optional)
  std::unique_ptr<char[]> buffer{};
};

NPYArray load_npy(const char*, size_t);
NPYArray load_npy_compressed(const char*, uint32_t, uint32_t);

} // namespace spdl::archive
