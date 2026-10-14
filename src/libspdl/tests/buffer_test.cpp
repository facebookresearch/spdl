/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/buffer.h>

#include <libspdl/core/storage.h>
#include <libspdl/core/types.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace spdl::core {
namespace {

TEST(CPUBufferTest, AllocatedStorageCoversShapeTimesDepth) {
  auto buffer = cpu_buffer({2, 3, 4}, ElemClass::Float, sizeof(float));

  ASSERT_NE(buffer->storage, nullptr);
  EXPECT_EQ(buffer->storage->size, 2 * 3 * 4 * sizeof(float));
  EXPECT_EQ(buffer->shape, std::vector<size_t>({2, 3, 4}));
  EXPECT_EQ(buffer->elem_class, ElemClass::Float);
  EXPECT_EQ(buffer->depth, sizeof(float));
}

TEST(CPUBufferTest, ScalarShapeAllocatesSingleElement) {
  auto buffer = cpu_buffer({}, ElemClass::Int, sizeof(int32_t));

  EXPECT_EQ(buffer->storage->size, sizeof(int32_t));
}

TEST(CPUBufferTest, DataIsWritableAcrossWholeBuffer) {
  auto buffer = cpu_buffer({8, 8});
  auto* data = static_cast<uint8_t*>(buffer->data());
  ASSERT_NE(data, nullptr);

  for (size_t i = 0; i < 64; ++i) {
    data[i] = static_cast<uint8_t>(i);
  }

  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[63], 63);
}

TEST(CPUBufferTest, ConstDataReturnsSamePointerAsMutableData) {
  auto buffer = cpu_buffer({4});
  const CPUBuffer& const_buffer = *buffer;

  EXPECT_EQ(const_buffer.data(), buffer->data());
}

TEST(CPUBufferTest, ProvidedStorageIsReusedInsteadOfAllocating) {
  auto storage = std::make_shared<CPUStorage>(128);
  void* storage_data = storage->data();

  auto buffer = cpu_buffer({4, 4}, ElemClass::UInt, sizeof(uint8_t), storage);

  EXPECT_EQ(buffer->storage, storage);
  EXPECT_EQ(buffer->data(), storage_data);
  EXPECT_EQ(buffer->storage->size, 128);
}

TEST(CPUBufferTest, ProvidedStorageOfExactSizeIsAccepted) {
  auto storage = std::make_shared<CPUStorage>(16);

  auto buffer = cpu_buffer({4, 4}, ElemClass::UInt, sizeof(uint8_t), storage);

  EXPECT_EQ(buffer->data(), storage->data());
}

TEST(CPUBufferTest, ProvidedStorageSmallerThanRequestedThrows) {
  auto storage = std::make_shared<CPUStorage>(15);

  EXPECT_THROW(
      {
        auto buffer =
            cpu_buffer({4, 4}, ElemClass::UInt, sizeof(uint8_t), storage);
      },
      std::runtime_error);
}

TEST(CPUBufferTest, ProvidedStorageTooSmallForDepthThrows) {
  auto storage = std::make_shared<CPUStorage>(16);

  EXPECT_THROW(
      {
        auto buffer =
            cpu_buffer({4, 4}, ElemClass::Float, sizeof(float), storage);
      },
      std::runtime_error);
}

} // namespace
} // namespace spdl::core
