/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include <gtest/gtest.h>

#include <utility>

namespace spdl::core {
namespace {

TEST(PacketsTest, DefaultConstructionInitializesStreamIndex) {
  VideoPackets packets;
  EXPECT_EQ(packets.stream_index, 0);
}

TEST(PacketsTest, CopyPreservesStreamIndex) {
  VideoPackets source("memory://video", 7, Rational{1, 90'000});
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  VideoPackets copied(source);
  EXPECT_EQ(copied.stream_index, 7);

  VideoPackets assigned;
  assigned = source;
  EXPECT_EQ(assigned.stream_index, 7);
}

TEST(PacketsTest, MovePreservesStreamIndex) {
  VideoPackets source("memory://video", 7, Rational{1, 90'000});

  VideoPackets moved(std::move(source));
  EXPECT_EQ(moved.stream_index, 7);

  VideoPackets assigned("memory://other", 3, Rational{1, 1'000});
  assigned = std::move(moved);
  EXPECT_EQ(assigned.stream_index, 7);
}

} // namespace
} // namespace spdl::core
