/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/bytes.h"

#include <gtest/gtest.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace spdl::core::detail {
namespace {

TEST(BytesSeekTest, SupportsNegativeRelativeOffsets) {
  size_t position = 5;

  EXPECT_EQ(seek_bytes(10, position, -3, SEEK_CUR), 2);
  EXPECT_EQ(position, 2);
  EXPECT_EQ(seek_bytes(10, position, -2, SEEK_END), 8);
  EXPECT_EQ(position, 8);
}

TEST(BytesSeekTest, SupportsForcedSeeksAndSizeQueries) {
  size_t position = 3;

  EXPECT_EQ(seek_bytes(10, position, 4, SEEK_SET | AVSEEK_FORCE), 4);
  EXPECT_EQ(position, 4);
  EXPECT_EQ(seek_bytes(10, position, 2, SEEK_CUR | AVSEEK_FORCE), 6);
  EXPECT_EQ(position, 6);
  EXPECT_EQ(seek_bytes(10, position, 0, AVSEEK_SIZE | AVSEEK_FORCE), 10);
  EXPECT_EQ(position, 6);
}

TEST(BytesSeekTest, RejectsOutOfRangeSeeksWithoutChangingPosition) {
  size_t position = 5;

  EXPECT_EQ(seek_bytes(10, position, -6, SEEK_CUR), AVERROR(EINVAL));
  EXPECT_EQ(position, 5);
  EXPECT_EQ(seek_bytes(10, position, -1, SEEK_SET), AVERROR(EINVAL));
  EXPECT_EQ(position, 5);
  EXPECT_EQ(seek_bytes(10, position, 11, SEEK_SET), AVERROR(EINVAL));
  EXPECT_EQ(position, 5);
  EXPECT_EQ(
      seek_bytes(10, position, std::numeric_limits<int64_t>::min(), SEEK_END),
      AVERROR(EINVAL));
  EXPECT_EQ(position, 5);
}

TEST(BytesSeekTest, RejectsSizesThatCannotFitInCallbackResult) {
  if constexpr (
      std::numeric_limits<size_t>::max() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    size_t position = 0;
    const size_t size =
        static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;
    EXPECT_EQ(seek_bytes(size, position, 0, AVSEEK_SIZE), AVERROR(EOVERFLOW));
  }
}

} // namespace
} // namespace spdl::core::detail
