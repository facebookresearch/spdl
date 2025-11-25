/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/storage.h>
#include <libspdl/cuda/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace spdl::cuda {
void* alloc_pinned(size_t s);
void dealloc_pinned(void* p);

class CUDAStorage : public core::Storage {
  void* data_ = nullptr;

 public:
  uintptr_t stream_ = 0;

  cuda_deleter_fn deleter_;

  void* data() const override;

  CUDAStorage() = default;
  CUDAStorage(size_t size, const CUDAConfig& cfg);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage() override;
};

using CUDAStoragePtr = std::shared_ptr<CUDAStorage>;

} // namespace spdl::cuda
