/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/utils.h>

#include <gtest/gtest.h>

extern "C" {
#include <libavutil/rational.h>
}

namespace spdl::core::detail {
namespace {

TEST(RationalUtilsTest, IsWithinWindowBasic) {
  AVRational val = {1, 2}; // 0.5
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  EXPECT_TRUE(is_within_window(val, start, end));
}

TEST(RationalUtilsTest, IsWithinWindowAtStart) {
  AVRational val = {1, 4}; // 0.25
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  // Should be true because start <= val
  EXPECT_TRUE(is_within_window(val, start, end));
}

TEST(RationalUtilsTest, IsWithinWindowAtEnd) {
  AVRational val = {3, 4}; // 0.75
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  // Should be false because val < end is required (half-open interval)
  EXPECT_FALSE(is_within_window(val, start, end));
}

TEST(RationalUtilsTest, IsWithinWindowHalfOpenRange) {
  // Explicitly test half-open range behavior: [start, end)
  // This means: start <= val < end
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  // Test lower boundary: val == start should be included
  EXPECT_TRUE(is_within_window(start, start, end));

  // Test upper boundary: val == end should be excluded
  EXPECT_FALSE(is_within_window(end, start, end));
}

TEST(RationalUtilsTest, IsWithinWindowBeforeStart) {
  AVRational val = {1, 8}; // 0.125
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  EXPECT_FALSE(is_within_window(val, start, end));
}

TEST(RationalUtilsTest, IsWithinWindowAfterEnd) {
  AVRational val = {7, 8}; // 0.875
  AVRational start = {1, 4}; // 0.25
  AVRational end = {3, 4}; // 0.75

  EXPECT_FALSE(is_within_window(val, start, end));
}

TEST(RationalUtilsTest, ToRationalBasic) {
  AVRational time_base = {1, 1000}; // 1 millisecond
  int64_t val = 5000; // 5000 milliseconds = 5 seconds

  AVRational result = to_rational(val, time_base);

  // Result should be 5/1 (5 seconds)
  EXPECT_EQ(result.num, 5);
  EXPECT_EQ(result.den, 1);
}

TEST(RationalUtilsTest, ToRationalWithReduction) {
  AVRational time_base = {1, 1000}; // 1 millisecond
  int64_t val = 2000; // 2000 milliseconds = 2 seconds

  AVRational result = to_rational(val, time_base);

  // Result should be reduced to 2/1 (2 seconds)
  EXPECT_EQ(result.num, 2);
  EXPECT_EQ(result.den, 1);
}

TEST(RationalUtilsTest, ToRationalWithLargeValue) {
  AVRational time_base = {1, 90000}; // Common video time base
  // Use a value outside of int32_t range
  int64_t val = 5000000000LL; // 5 billion, which is > INT32_MAX (2147483647)

  AVRational result = to_rational(val, time_base);

  // The result should be reduced, but still represent the same time value
  // 5000000000 / 90000 = 55555.555... seconds
  // After reduction with INT32_MAX constraint, we expect a rational that
  // approximates this value
  EXPECT_GT(result.num, 0);
  EXPECT_GT(result.den, 0);

  // Verify the rational approximates the correct time value
  double expected_time =
      static_cast<double>(val * time_base.num) / time_base.den;
  double actual_time = static_cast<double>(result.num) / result.den;

  // Allow for some error due to reduction
  EXPECT_NEAR(expected_time, actual_time, 1.0); // Within 1 second tolerance
}

TEST(RationalUtilsTest, ToRationalZeroValue) {
  AVRational time_base = {1, 1000};
  int64_t val = 0;

  AVRational result = to_rational(val, time_base);

  EXPECT_EQ(result.num, 0);
  EXPECT_GT(result.den, 0); // Denominator should be positive
}

TEST(RationalUtilsTest, ToRationalNegativeValue) {
  AVRational time_base = {1, 1000};
  int64_t val = -5000;

  AVRational result = to_rational(val, time_base);

  // Result should be -5/1
  EXPECT_LT(result.num, 0);
  EXPECT_GT(result.den, 0);
  EXPECT_EQ(result.num, -5);
  EXPECT_EQ(result.den, 1);
}

} // namespace
} // namespace spdl::core::detail
