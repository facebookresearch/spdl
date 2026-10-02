/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <gtest/gtest.h>

#include <utility>

namespace spdl::core::detail {
namespace {

struct Tracked {
  int num_releases = 0;
};

void release_tracked(Tracked** ptr) {
  if (*ptr) {
    ++(*ptr)->num_releases;
    *ptr = nullptr;
  }
}

using TrackedDPtr = DPtr<Tracked, release_tracked>;

TEST(DPtrTest, MoveConstructionTransfersOwnership) {
  Tracked tracked;
  {
    TrackedDPtr source(&tracked);
    Tracked* const raw = source;

    TrackedDPtr destination(std::move(source));

    EXPECT_EQ(
        // NOLINTNEXTLINE(bugprone-use-after-move)
        static_cast<Tracked*>(source),
        nullptr);
    EXPECT_EQ(static_cast<Tracked*>(destination), raw);
  }
  EXPECT_EQ(tracked.num_releases, 1);
}

TEST(DPtrTest, MoveAssignmentReleasesOldObjectAndTransfersOwnership) {
  Tracked source_tracked;
  Tracked destination_tracked;
  {
    TrackedDPtr source(&source_tracked);
    Tracked* const source_raw = source;
    TrackedDPtr destination(&destination_tracked);

    destination = std::move(source);

    EXPECT_EQ(destination_tracked.num_releases, 1);
    EXPECT_EQ(
        // NOLINTNEXTLINE(bugprone-use-after-move)
        static_cast<Tracked*>(source),
        nullptr);
    EXPECT_EQ(static_cast<Tracked*>(destination), source_raw);
  }
  EXPECT_EQ(source_tracked.num_releases, 1);
  EXPECT_EQ(destination_tracked.num_releases, 1);
}

} // namespace
} // namespace spdl::core::detail
