/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frames.h>

#include <gtest/gtest.h>

#include <utility>

namespace spdl::core {
namespace {

TEST(FramesTest, MoveConstructionPreservesTimeBase) {
  VideoFrames source(17, Rational{1, 90'000});

  VideoFrames destination(std::move(source));

  EXPECT_EQ(destination.get_id(), 17);
  EXPECT_EQ(destination.get_time_base().num, 1);
  EXPECT_EQ(destination.get_time_base().den, 90'000);
}

TEST(FramesTest, MoveAssignmentPreservesTimeBase) {
  VideoFrames source(17, Rational{1, 90'000});
  VideoFrames destination(23, Rational{1, 1'000});

  destination = std::move(source);

  EXPECT_EQ(destination.get_id(), 17);
  EXPECT_EQ(destination.get_time_base().num, 1);
  EXPECT_EQ(destination.get_time_base().den, 90'000);
}

} // namespace
} // namespace spdl::core
