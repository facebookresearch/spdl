/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frame_arena.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

namespace spdl::core {
namespace {

struct Header {
  size_t total_size;
  uint64_t flags;
  char padding[16];
};

static_assert(sizeof(Header) == FrameArena::kAlign);
static_assert(alignof(Header) <= FrameArena::kAlign);

constexpr uint64_t kFlagDirectAlloc = 1;

size_t round_up(size_t size, size_t granularity) {
  return ((size + granularity - 1) / granularity) * granularity;
}

} // namespace

FrameArena::FrameArena(Config config) : config_(config) {
  auto initial = std::make_unique<char[]>(config_.initial_size);
  slabs_.push_back(std::move(initial));
  current_slab_capacity_ = config_.initial_size;
  slab_offset_ = 0;
  total_allocated_ = config_.initial_size;
}

FrameArena::~FrameArena() = default;

FrameArena::ThreadLocal& FrameArena::get_thread_local() {
  thread_local ThreadLocal tl;
  return tl;
}

void* FrameArena::allocate(size_t size) {
  size_t rounded = round_up(size, config_.granularity);
  size_t total = rounded + kAlign;

  auto& tl = get_thread_local();
  auto it = tl.free_lists.find(total);
  if (it != tl.free_lists.end() && !it->second.empty()) {
    void* raw = it->second.back();
    it->second.pop_back();
    return static_cast<char*>(raw) + kAlign;
  }

  void* raw = allocate_from_slab(total);
  if (!raw) {
    return nullptr;
  }

  return static_cast<char*>(raw) + kAlign;
}

void* FrameArena::allocate_from_slab(size_t total) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (slab_offset_ + total <= current_slab_capacity_) {
    void* ptr = slabs_.back().get() + slab_offset_;
    slab_offset_ += total;
    auto* header = static_cast<Header*>(ptr);
    header->total_size = total;
    header->flags = 0;
    std::memset(header->padding, 0, sizeof(header->padding));
    return ptr;
  }

  size_t new_slab_size = std::max(config_.initial_size, total);

  if (total_allocated_ + new_slab_size > config_.max_size) {
    void* raw = operator new(total, std::nothrow);
    if (!raw) {
      return nullptr;
    }
    auto* header = static_cast<Header*>(raw);
    header->total_size = total;
    header->flags = kFlagDirectAlloc;
    std::memset(header->padding, 0, sizeof(header->padding));
    return raw;
  }

  auto new_slab = std::make_unique<char[]>(new_slab_size);
  void* ptr = new_slab.get();
  slabs_.push_back(std::move(new_slab));
  current_slab_capacity_ = new_slab_size;
  slab_offset_ = total;
  total_allocated_ += new_slab_size;

  auto* header = static_cast<Header*>(ptr);
  header->total_size = total;
  header->flags = 0;
  std::memset(header->padding, 0, sizeof(header->padding));
  return ptr;
}

void FrameArena::deallocate(void* user_ptr) {
  if (!user_ptr) {
    return;
  }

  void* raw = static_cast<char*>(user_ptr) - kAlign;
  auto* header = static_cast<Header*>(raw);
  size_t total = header->total_size;

  if (header->flags & kFlagDirectAlloc) {
    operator delete(raw);
    return;
  }

  auto& tl = get_thread_local();
  auto& bucket = tl.free_lists[total];
  if (bucket.size() < config_.max_free_per_bucket) {
    bucket.push_back(raw);
  }
}

size_t FrameArena::total_allocated() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_allocated_;
}

size_t FrameArena::total_pooled() const {
  auto& tl = get_thread_local();
  size_t count = 0;
  for (const auto& [size, ptrs] : tl.free_lists) {
    count += ptrs.size();
  }
  return count;
}

} // namespace spdl::core
