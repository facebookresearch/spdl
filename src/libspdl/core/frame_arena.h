/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace spdl::core {

class FrameArena {
 public:
  static constexpr size_t kAlign = 32;

  struct Config {
    size_t initial_size = 128 * 1024 * 1024;
    size_t max_size = 1024 * 1024 * 1024;
    size_t granularity = 64 * 1024;
    size_t max_free_per_bucket = 8;
  };

  explicit FrameArena(Config config);
  ~FrameArena();

  FrameArena(const FrameArena&) = delete;
  FrameArena& operator=(const FrameArena&) = delete;

  void* allocate(size_t size);
  void deallocate(void* ptr);

  size_t total_allocated() const;
  size_t total_pooled() const;

 private:
  struct ThreadLocal {
    std::unordered_map<size_t, std::vector<void*>> free_lists;
  };

  static ThreadLocal& get_thread_local();

  void* allocate_from_slab(size_t total);

  Config config_;

  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<char[]>> slabs_;
  size_t current_slab_capacity_ = 0;
  size_t slab_offset_ = 0;
  size_t total_allocated_ = 0;
};

using FrameArenaPtr = std::shared_ptr<FrameArena>;

} // namespace spdl::core
