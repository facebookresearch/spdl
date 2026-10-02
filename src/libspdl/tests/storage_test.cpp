/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/storage.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

namespace spdl::core {
namespace {

int& deallocations_a() {
  static int value = 0;
  return value;
}

int& deallocations_b() {
  static int value = 0;
  return value;
}

void* allocate(size_t size) {
  return ::operator new(size);
}

void deallocate_a(void* ptr) {
  ++deallocations_a();
  ::operator delete(ptr);
}

void deallocate_b(void* ptr) {
  ++deallocations_b();
  ::operator delete(ptr);
}

TEST(CPUStorageTest, DefaultConstructionIsSafe) {
  CPUStorage storage;

  EXPECT_EQ(storage.data(), nullptr);
  EXPECT_EQ(storage.size, 0);
  EXPECT_FALSE(storage.is_pinned());
}

TEST(CPUStorageTest, MoveConstructionTransfersAllState) {
  deallocations_a() = 0;
  void* data = nullptr;
  {
    CPUStorage source(32, allocate, deallocate_a, true);
    data = source.data();

    CPUStorage destination(std::move(source));

    EXPECT_EQ(destination.data(), data);
    EXPECT_EQ(destination.size, 32);
    EXPECT_TRUE(destination.is_pinned());
    EXPECT_EQ(source.data(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.size, 0); // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(source.is_pinned()); // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(deallocations_a(), 1);
}

TEST(CPUStorageTest, MoveAssignmentTransfersAllStateAndOwnership) {
  deallocations_a() = 0;
  deallocations_b() = 0;
  {
    CPUStorage source(32, allocate, deallocate_a, true);
    CPUStorage destination(8, allocate, deallocate_b, false);
    void* source_data = source.data();

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination.size, 32);
    EXPECT_TRUE(destination.is_pinned());
  }
  EXPECT_EQ(deallocations_a(), 1);
  EXPECT_EQ(deallocations_b(), 1);
}

} // namespace
} // namespace spdl::core
