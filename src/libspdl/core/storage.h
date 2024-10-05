/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#ifdef SPDL_USE_CUDA
#include <cuda.h>
#endif

namespace spdl::core {

struct Storage {
  virtual void* data() const = 0;
  virtual ~Storage() = default;
};

class CPUStorage : public Storage {
 public:
  size_t size;
  // So far, we only need this in CPUStorage. So we are not adding it
  // in CUDAStorage. If we need to add that to CUDAStorage, revisit
  // the interface/abstraction. (Is virtual `get_size` better?)
 private:
  void* data_ = nullptr;
  bool memory_pinned = false;

 public:
  void* data() const override;
  bool is_pinned() const;

  CPUStorage() = default;
  explicit CPUStorage(size_t size, bool pin_memory = false);

  CPUStorage(const CPUStorage&) = delete;
  CPUStorage& operator=(const CPUStorage&) = delete;

  CPUStorage(CPUStorage&&) noexcept;
  CPUStorage& operator=(CPUStorage&&) noexcept;

  ~CPUStorage() override;
};

class CUDAStorage : public Storage {
#ifdef SPDL_USE_CUDA
  void* data_ = nullptr;

 public:
  CUstream stream = nullptr;

  cuda_deleter_fn deleter;

  void* data() const override;

  CUDAStorage() = default;
  CUDAStorage(size_t size, int device, CUstream stream);
  CUDAStorage(size_t size, const CUDAConfig& cfg);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage() override;
#endif
};

} // namespace spdl::core
